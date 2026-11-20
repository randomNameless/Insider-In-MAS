#!/usr/bin/env bash
# Defense thread (2026-11-09, "ok let's do dpo"): DPO variant of the adversarially trained honest victim.
#   1. prompts that got no consistent candidate in the K=4 pass (7,153) get EXTRA_K more samples (seed tag "x8")
#   2. build_dpo_pairs: chosen = consistent candidate, rejected = swayed candidate (or the swayed trajectory reply)
#   3. at_dpo_train (LoRA r16, beta 0.1, 2 epochs) -> export_vllm -> ~/models/at-dpo-9b
#   4. eval test 400 x seeds 3/4/5: none / sd120 / sd120+dwm  (dirs at_dpo_{none,sd120,dwmbon}), paired with cons + undefended
# Smoke: SMOKE=1 (EXTRA_K sampling limited to 200 prompts, max 1 epoch, N=2 eval; dirs at_smoke_dpo_*).
# setsid nohup bash ~/consensus-attacker/debate_wm/at_dpo_chain_lambda.sh > ~/runs/at_dpo_chain.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
M=/path/to/models; R=/path/to/runs; DATA=/path/to/data/at; PS=v1nn
TASKS=/path/to/data/mmlupro_v1/tasks.jsonl
TGT=$R/targets_none9b_exp10_nn.json
RSD=$R/exp9_debate_ppo_v6_stratdpo_base
CK="sd120=$RSD/checkpoint-0120 (merged: exp9-v6sd-tgt-c120)"
EXTRA_K=${EXTRA_K:-8}
if [ -n "${SMOKE:-}" ]; then LIMIT="--limit 200"; EPOCHS=1; N=2; SEEDS=3; TAG=at_smoke; V=dpo; D=$DATA/smoke_dpo; else LIMIT=""; EPOCHS=2; N=400; SEEDS=3,4,5; TAG=at; V=dpo; D=$DATA; fi
mkdir -p $D
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

ts "step1 prompts without a consistent candidate"
[ -f $D/fit_attacked_nochosen.jsonl ] || python -m debate_wm.build_dpo_pairs --attacked $DATA/fit_attacked.jsonl \
  --cands $DATA/fit_cons_cands.jsonl --missing-out $D/fit_attacked_nochosen.jsonl
ts "step1b extra K=$EXTRA_K samples for them"
[ -f $D/fit_cons_cands_x8.jsonl ] || python -m debate_wm.at_sample --model $M/qwen3.5-9b --in $D/fit_attacked_nochosen.jsonl \
  --out $D/fit_cons_sft_x8.jsonl --cands-out $D/fit_cons_cands_x8.jsonl --target cons --k $EXTRA_K --seed-tag x8 $LIMIT
ts "step2 build pairs"
[ -f $D/train_dpo.jsonl ] || python -m debate_wm.build_dpo_pairs --attacked $DATA/fit_attacked.jsonl \
  --cands $DATA/fit_cons_cands.jsonl $D/fit_cons_cands_x8.jsonl --train $D/train_dpo.jsonl --val $D/val_dpo.jsonl
LORA=$R/${TAG}_${V}_lora
ts "step3 DPO -> $LORA"
[ -f $LORA/merged_text/at_meta.json ] || python -m debate_wm.at_dpo_train --model $M/qwen3.5-9b --train $D/train_dpo.jsonl \
  --val $D/val_dpo.jsonl --out $LORA --merge-out $LORA/merged_text --epochs $EPOCHS > $LORA.log 2>&1
if [ ! -f $M/${TAG}-${V}-9b/at_meta.json ]; then
  python -m debate_wm.export_vllm --trained $LORA/merged_text --base $M/qwen3.5-9b --out $M/${TAG}-${V}-9b
  cp $LORA/merged_text/at_meta.json $M/${TAG}-${V}-9b/at_meta.json
  rm -rf $LORA/merged_text/*.safetensors
fi
VICTIM=$M/${TAG}-${V}-9b
COMMON="--victim $VICTIM --seeds $SEEDS --n $N --answer-first --profile-set $PS --tasks $TASKS --arg-tokens 200 --targets $TGT"
RUN=$R/${TAG}_${V}_none; ts "step4 eval none -> $RUN"
[ -f $RUN/state.json ] || python -m debate_ppo.eval_debate init --run $RUN --label "${TAG}_${V}_none_honest_suggestible_neutral_neutral" --attackers '{}' --conds none $COMMON
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
RUN=$R/${TAG}_${V}_sd120; ts "step4 eval sd120 -> $RUN"
[ -f $RUN/state.json ] || python -m debate_ppo.eval_debate init --run $RUN --label "${TAG}_${V}_sd120_honest_suggestible_neutral_neutral" \
  --attackers "{\"targeted_model_rl_sd120\":\"$M/exp9-v6sd-tgt-c120\"}" --conds targeted_model_rl_sd120 --rl-checkpoint "$CK" $COMMON
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
RUN=$R/${TAG}_${V}_dwmbon; ts "step4 eval dwmbon -> $RUN"
[ -f $RUN/state.json ] || python -m debate_ppo.eval_debate_dwm init --run $RUN --label "${TAG}_${V}_dwmbon_honest_suggestible_neutral_neutral" \
  --attackers "{\"targeted_model_rl_sd120__wmbon8dw\":\"$M/exp9-v6sd-tgt-c120\"}" --conds targeted_model_rl_sd120__wmbon8dw \
  --rl-checkpoint "$CK" --gpu-util 0.6 --dwm-model $M/debate-wm-v1 $COMMON
python -m debate_ppo.eval_debate_dwm play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
[ -n "${SMOKE:-}" ] || python debate_wm/def_paired.py --tag at --defs cons,dpo > $R/at_paired.md 2> $R/at_paired.err || true
ts "AT_DPO_CHAIN_DONE"
