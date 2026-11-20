#!/usr/bin/env bash
# No-peek selector (2026-11-12): the dwm simulates the honest agents' round-r replies instead of reading the real
# ones (debate_ppo/eval_debate_dwm_np.py). Flags identical to exp10_nn_dwmbon3k_* / exp10_nn_ctl_* (only the selector's
# round-r context changes). Paired against exp10_nn_{none,ours,dwmbon,ctl_ju,ctl_kraidia}.
#   WHICH=ours     run exp10_nn_np_ours:    targeted_model_rl_sd120__wmbon8dw
#   WHICH=ju       run exp10_nn_np_ju:      ju_dpo_q3_4b_wmbon8dw
#   WHICH=kraidia  run exp10_nn_np_kraidia: kraidia_4b_wmbon8dw
# setsid nohup env WHICH=ours bash debate_wm/np_selector_lambda.sh > ~/runs/np_selector_ours.log 2>&1 &
# SMOKE=1: N=2 SEEDS=3, *_smoke dirs.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
M=/path/to/models; R=/path/to/runs; TAG=exp10_nn; RSD=$R/exp9_debate_ppo_v6_stratdpo_base
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
S=${SMOKE:+_smoke}
if [ -n "$S" ]; then SEEDS=3; N=2; else SEEDS=3,4,5; N=400; fi
COMMON="--victim $M/qwen3.5-9b --seeds $SEEDS --n $N --answer-first --profile-set v1nn --tasks /path/to/data/mmlupro_v1/tasks.jsonl --gpu-util 0.6 --targets $R/targets_none9b_${TAG}.json --dwm-model $M/debate-wm-v1"
run() {  # name cond model extra...
  local name=$1 cond=$2 model=$3; shift 3
  local RUN=$R/${TAG}_np_${name}$S
  if [ ! -f $RUN/state.json ]; then
    python -m debate_ppo.eval_debate_dwm_np init --run $RUN --label "${TAG}_np_${name}_honest_suggestible_neutral_neutral" \
      --attackers "{\"$cond\":\"$model\"}" --conds $cond "$@" $COMMON
  fi
  ts "START np_$name -> $RUN"
  python -m debate_ppo.eval_debate_dwm_np play --run $RUN >> $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
  ts "DONE np_$name"
}
case $WHICH in
  ours)    run ours targeted_model_rl_sd120__wmbon8dw $M/exp9-v6sd-tgt-c120 --arg-tokens 200 \
             --rl-checkpoint "sd120=$RSD/checkpoint-0120 (merged: exp9-v6sd-tgt-c120)" ;;
  ju)      run ju ju_dpo_q3_4b_wmbon8dw $M/ju-dpo-q3-4b --arg-tokens 200 --max-model-len 16384 ;;
  kraidia) run kraidia kraidia_4b_wmbon8dw $M/qwen3-4b --max-model-len 16384 ;;
esac
ts "NP_SELECTOR_DONE $WHICH"
