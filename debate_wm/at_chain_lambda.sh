#!/usr/bin/env bash
# Defense thread (2026-11-08): adversarial training of the honest victim, end to end.
#   1. build_at_data : attacked honest prompts on FIT (def_fit_sd120 + def_fit_dwmbon) + clean targets from stratdpo6_collect_v1nn_fit
#   2. at_sample     : K=4 rejection sampling with Qwen3.5-9B (target = cons | gold)
#   3. mix           : accepted attacked records + clean anchor records (ratio CLEAN_FRAC of the total), split by task id
#                      into train / val (val = tasks whose id hash falls in the last 10%; the TEST 400 are never used)
#   4. at_train      : LoRA r=16, 2 epochs, merge -> ~/models/at-<variant>-9b
#   5. eval          : test 400 x seeds 3/4/5, none / sd120 / sd120+dwm with the merged model as --victim
#                      (dirs at_<variant>_{none,sd120,dwmbon}), paired against exp10_nn_* by def_paired.py
# VARIANTS=cons (default) or cons,gold.   Smoke: SMOKE=1 (limits sampling to 300 prompts, 1 epoch, N=2 eval).
# setsid nohup bash ~/consensus-attacker/debate_wm/at_chain_lambda.sh > ~/runs/at_chain.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
M=/path/to/models; R=/path/to/runs; DATA=/path/to/data/at; PS=v1nn
TASKS=/path/to/data/mmlupro_v1/tasks.jsonl
TGT=$R/targets_none9b_exp10_nn.json
RSD=$R/exp9_debate_ppo_v6_stratdpo_base
CK="sd120=$RSD/checkpoint-0120 (merged: exp9-v6sd-tgt-c120)"
VARIANTS=${VARIANTS:-cons}; CLEAN_FRAC=${CLEAN_FRAC:-0.3}; K=${K:-4}
if [ -n "${SMOKE:-}" ]; then LIMIT="--limit 300"; EPOCHS=1; N=2; SEEDS=3; TAG=at_smoke; DATA=$DATA/smoke; else LIMIT=""; EPOCHS=2; N=400; SEEDS=3,4,5; TAG=at; fi
mkdir -p $DATA
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

ts "step1 build_at_data"
[ -f $DATA/fit_attacked.jsonl ] || python -m debate_wm.build_at_data --attack $R/def_fit_sd120/state.json $R/def_fit_dwmbon/state.json \
  --clean $R/stratdpo6_collect_v1nn_fit/state.json --out $DATA/fit_attacked.jsonl --clean-out $DATA/fit_clean.jsonl

for V in ${VARIANTS//,/ }; do
  ts "step2 at_sample $V"
  [ -f $DATA/fit_${V}_sft.jsonl ] || python -m debate_wm.at_sample --model $M/qwen3.5-9b --in $DATA/fit_attacked.jsonl \
    --out $DATA/fit_${V}_sft.jsonl --cands-out $DATA/fit_${V}_cands.jsonl --target $V --k $K $LIMIT
  ts "step3 mix/split $V"
  python - "$DATA" "$V" "$CLEAN_FRAC" <<'EOF'
import hashlib, json, random, sys
data, v, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
att = [json.loads(l) for l in open(f"{data}/fit_{v}_sft.jsonl")]
clean = [json.loads(l) for l in open(f"{data}/fit_clean.jsonl")]
tasks = {r["meta"]["task_id"] for r in att} | {r["meta"]["task_id"] for r in clean}
assert all(r["meta"]["task_id"].startswith("mmlupro-") for r in att)
val_tasks = {t for t in tasks if int(hashlib.sha256(("at_val|" + t).encode()).hexdigest()[:8], 16) % 10 == 9}
rng = random.Random(0)
n_clean = int(round(len(att) * frac / (1 - frac)))
rng.shuffle(clean); clean = clean[:n_clean]
tr = open(f"{data}/train_{v}.jsonl", "w"); va = open(f"{data}/val_{v}.jsonl", "w")
cnt = {"train_att": 0, "train_clean": 0, "val_att": 0, "val_clean": 0}
for r in att + clean:
    is_val = r["meta"]["task_id"] in val_tasks
    kind = "att" if r["meta"]["source"].startswith("rs_") else "clean"
    (va if is_val else tr).write(json.dumps(r) + "\n"); cnt[("val_" if is_val else "train_") + kind] += 1
print(json.dumps({"variant": v, "clean_frac": frac, "n_val_tasks": len(val_tasks), **cnt}))
json.dump({"variant": v, "clean_frac": frac, "val_tasks": sorted(val_tasks), **cnt}, open(f"{data}/split_{v}.json", "w"), indent=1)
EOF
  ts "step4 at_train $V -> $R/${TAG}_${V}_lora, merge -> $M/${TAG}-${V}-9b"
  # at_train merges the adapter into the TEXT model (Qwen3_5ForCausalLM); export_vllm re-attaches the original config,
  # vision tower and MTP weights so vLLM can serve it (same path as debate-wm-v1).
  LORA=$R/${TAG}_${V}_lora
  [ -f $LORA/merged_text/at_meta.json ] || python -m debate_wm.at_train --model $M/qwen3.5-9b --train $DATA/train_${V}.jsonl \
    --val $DATA/val_${V}.jsonl --out $LORA --merge-out $LORA/merged_text --epochs $EPOCHS > $LORA.log 2>&1
  if [ ! -f $M/${TAG}-${V}-9b/at_meta.json ]; then
    python -m debate_wm.export_vllm --trained $LORA/merged_text --base $M/qwen3.5-9b --out $M/${TAG}-${V}-9b
    cp $LORA/merged_text/at_meta.json $M/${TAG}-${V}-9b/at_meta.json
    rm -rf $LORA/merged_text/*.safetensors      # 18 GB duplicate of the exported weights
  fi
  VICTIM=$M/${TAG}-${V}-9b
  COMMON="--victim $VICTIM --seeds $SEEDS --n $N --answer-first --profile-set $PS --tasks $TASKS --arg-tokens 200 --targets $TGT"

  RUN=$R/${TAG}_${V}_none; ts "step5 eval none -> $RUN"
  [ -f $RUN/state.json ] || python -m debate_ppo.eval_debate init --run $RUN --label "${TAG}_${V}_none_honest_suggestible_neutral_neutral" \
    --attackers '{}' --conds none $COMMON
  python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1

  RUN=$R/${TAG}_${V}_sd120; ts "step5 eval sd120 -> $RUN"
  [ -f $RUN/state.json ] || python -m debate_ppo.eval_debate init --run $RUN --label "${TAG}_${V}_sd120_honest_suggestible_neutral_neutral" \
    --attackers "{\"targeted_model_rl_sd120\":\"$M/exp9-v6sd-tgt-c120\"}" --conds targeted_model_rl_sd120 --rl-checkpoint "$CK" $COMMON
  python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1

  RUN=$R/${TAG}_${V}_dwmbon; ts "step5 eval dwmbon -> $RUN"
  [ -f $RUN/state.json ] || python -m debate_ppo.eval_debate_dwm init --run $RUN --label "${TAG}_${V}_dwmbon_honest_suggestible_neutral_neutral" \
    --attackers "{\"targeted_model_rl_sd120__wmbon8dw\":\"$M/exp9-v6sd-tgt-c120\"}" --conds targeted_model_rl_sd120__wmbon8dw \
    --rl-checkpoint "$CK" --gpu-util 0.6 --dwm-model $M/debate-wm-v1 $COMMON
  python -m debate_ppo.eval_debate_dwm play --run $RUN >> $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
  ts "DONE $V"
done
[ -n "${SMOKE:-}" ] || python debate_wm/def_paired.py --tag at --defs ${VARIANTS} > $R/at_paired.md 2> $R/at_paired.err || true
ts "AT_CHAIN_DONE"
