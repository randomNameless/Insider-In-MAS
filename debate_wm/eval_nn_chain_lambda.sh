#!/usr/bin/env bash
# Re-run of the whole aligned test table with a new honest group (2026-10-27): profile set v1nn =
# suggestible + neutral + neutral (no stubborn agent), honest = Qwen3.5-9B, MMLU-Pro test n=400, seeds 3,4,5.
# Targets are re-derived from this group's own `none` round 1 (debate_wm/make_targets.py, same rule as targets_none9b.json).
# Runs (all paired against ${TAG}_none by task/seed/seat): none -> ours -> dpo -> lit -> wmbon.
# Smoke: TAG=smoke_nn N=2 SEEDS=3 SMOKE=1 $0
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models; R=/path/to/runs; PS=v1nn
TAG=${TAG:-exp10_nn}; N=${N:-400}; SEEDS=${SEEDS:-3,4,5}
TASKS=/path/to/data/mmlupro_v1/tasks.jsonl
TGT=$R/targets_none9b_${TAG}.json
R6B=/path/to/runs/exp9_debate_ppo_v6b_targeted_len200; RSD=/path/to/runs/exp9_debate_ppo_v6_stratdpo_base
run() {  # name conds attackers [extra init args]
  local name=$1 conds=$2 att=$3; shift 3
  local RUN=$R/${TAG}_$name
  echo "[$(date -u +%H:%M:%S)] START $name conds=$conds"
  if [ ! -f $RUN/state.json ]; then
    python -m debate_ppo.eval_debate init --run $RUN --label "${TAG}_${name}_honest_suggestible_neutral_neutral" \
      --victim $M/qwen3.5-9b --attackers "$att" --conds $conds --seeds $SEEDS --n $N --answer-first --profile-set $PS --tasks $TASKS "$@"
  fi
  python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
  echo "[$(date -u +%H:%M:%S)] DONE $name"
}
if [ -n "${SMOKE:-}" ]; then
  C_OURS=targeted_model_rl_sd120,targeted_4b; C_DPO=stratdpo_q3_4b,ju_dpo_q3_4b; C_LIT=amayuelasbon_4b,bozdag_q25_7b,tomap_3b; C_WM=targeted_model_rl_c120__wmbon8wb
else
  C_OURS=targeted_model_rl_sd120,targeted_model_rl_c120,targeted_model_base,targeted_4b
  C_DPO=stratdpo_q3_4b,strat_q3_4b,strat_q3_4b__D1,ju_dpo_q3_4b,ju_q3_4b,ju_q3_4b__fluency_effect
  C_LIT=amayuelas_4b,amayuelasbon_4b,kraidia_4b,kraidiabon_4b,bozdag_q25_7b,tomap_3b
  C_WM=targeted_model_rl_sd120__wmbon8wb,targeted_model_rl_c120__wmbon8wb
fi
run none none '{}'
[ -f $TGT ] || python -m debate_wm.make_targets $R/${TAG}_none/state.json --out $TGT
run ours $C_OURS "{\"targeted_model_rl_sd120\":\"$M/exp9-v6sd-tgt-c120\",\"targeted_model_rl_c120\":\"$M/exp9-v6-tgt-c120\",\"targeted_model_base\":\"$M/qwen3-4b\",\"targeted_4b\":\"$M/qwen3-4b\"}" \
  --arg-tokens 200 --targets $TGT --rl-checkpoint "sd120=$RSD/checkpoint-0120 (merged: exp9-v6sd-tgt-c120) c120=$R6B/checkpoint-0120 (merged: exp9-v6-tgt-c120)"
run dpo $C_DPO "{\"stratdpo_q3_4b\":\"$M/strat-dpo-q3-4b\",\"strat_q3_4b\":\"$M/qwen3-4b\",\"strat_q3_4b__D1\":\"$M/qwen3-4b\",\"ju_dpo_q3_4b\":\"$M/ju-dpo-q3-4b\",\"ju_q3_4b\":\"$M/qwen3-4b\",\"ju_q3_4b__fluency_effect\":\"$M/qwen3-4b\"}" \
  --max-model-len 16384 --arg-tokens 200 --strat-om-model $M/qwen3-4b --strat-max-tokens 400 --targets $TGT
run lit $C_LIT "{\"amayuelas_4b\":\"$M/qwen3-4b\",\"amayuelasbon_4b\":\"$M/qwen3-4b\",\"kraidia_4b\":\"$M/qwen3-4b\",\"kraidiabon_4b\":\"$M/qwen3-4b\",\"bozdag_q25_7b\":\"$M/qwen2.5-7b-instruct\",\"tomap_3b\":\"$M/qwen2.5-3b-instruct-tomap\"}" \
  --max-model-len 16384 --targets $TGT --tomap-encoder $M/bge-m3 --tomap-helper $M/qwen2.5-3b-instruct --tomap-predictor $M/tomap-predictor/best_mlp_model.pth
run wmbon $C_WM "{\"targeted_model_rl_sd120__wmbon8wb\":\"$M/exp9-v6sd-tgt-c120\",\"targeted_model_rl_c120__wmbon8wb\":\"$M/exp9-v6-tgt-c120\"}" \
  --arg-tokens 200 --gpu-util 0.6 --targets $TGT --rl-checkpoint "sd120=$RSD/checkpoint-0120 (merged: exp9-v6sd-tgt-c120) c120=$R6B/checkpoint-0120 (merged: exp9-v6-tgt-c120)"
python debate_wm/nn_paired.py $TAG > $R/${TAG}_paired.json 2> $R/${TAG}_paired.err || true
echo NN_CHAIN_DONE
