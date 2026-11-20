#!/usr/bin/env bash
# Same-selector control (2026-11-09, thread "budget"): published single-sample persuaders + debate-wm-v1 selector + 8
# candidates (debate_ppo/eval_debate_dwm_lit.py). exp10_nn harness, flags as the `dpo` / `lit` runs of
# eval_nn_chain_lambda.sh + the dwm flags of dwm_chain_lambda.sh; paired against exp10_nn_{none,dpo,lit,ours,dwmbon}.
#   run ctl_ju:      ju_dpo_q3_4b_wmbon8dw   (A)
#   run ctl_kraidia: kraidia_4b_wmbon8dw     (B)
# setsid nohup bash ~/consensus-attacker/debate_wm/ctl_selector_lambda.sh > ~/runs/ctl_selector.log 2>&1 &
# SMOKE=1: N=2 SEEDS=3, *_smoke dirs.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
M=/path/to/models; R=/path/to/runs; TAG=exp10_nn
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
S=${SMOKE:+_smoke}
if [ -n "$S" ]; then SEEDS=3; N=2; else SEEDS=3,4,5; N=400; fi
COMMON="--victim $M/qwen3.5-9b --seeds $SEEDS --n $N --answer-first --profile-set v1nn --tasks /path/to/data/mmlupro_v1/tasks.jsonl --max-model-len 16384 --gpu-util 0.6 --targets $R/targets_none9b_${TAG}.json --dwm-model $M/debate-wm-v1"
run() {  # name cond model extra...
  local name=$1 cond=$2 model=$3; shift 3
  local RUN=$R/${TAG}_ctl_${name}$S
  if [ ! -f $RUN/state.json ]; then
    python -m debate_ppo.eval_debate_dwm_lit init --run $RUN --label "${TAG}_ctl_${name}_honest_suggestible_neutral_neutral" \
      --attackers "{\"$cond\":\"$model\"}" --conds $cond "$@" $COMMON
  fi
  ts "START ctl_$name -> $RUN"
  python -m debate_ppo.eval_debate_dwm_lit play --run $RUN >> $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
  ts "DONE ctl_$name"
}
run ju ju_dpo_q3_4b_wmbon8dw $M/ju-dpo-q3-4b --arg-tokens 200
run kraidia kraidia_4b_wmbon8dw $M/qwen3-4b
ts "CTL_SELECTOR_DONE"
