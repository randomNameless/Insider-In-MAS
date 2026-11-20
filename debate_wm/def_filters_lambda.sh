#!/usr/bin/env bash
# Defense thread (2026-11-08): honest-side input filters (debate_ppo/defense.py) in the exp10_nn harness.
# For each defense D in wmsr, cpwbft, sac: runs none, sd120 (targeted_model_rl_sd120) and sd120+dwm
# (targeted_model_rl_sd120__wmbon8dw) on MMLU-Pro test 400 x seeds 3/4/5, v1nn, Qwen3.5-9B honest, same targets and
# seat/honest seeds as exp10_nn, so every debate pairs with exp10_nn_{none,ours,dwmbon} by (task, seed).
# Run dirs: ~/runs/def_<D>_{none,sd120,dwmbon}.   Smoke: SMOKE=1 (N=2, seed 3, dirs def_smoke_<D>_*).
# setsid nohup bash ~/consensus-attacker/debate_wm/def_filters_lambda.sh > ~/runs/def_filters.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
M=/path/to/models; R=/path/to/runs; PS=v1nn
TASKS=/path/to/data/mmlupro_v1/tasks.jsonl
TGT=$R/targets_none9b_exp10_nn.json
RSD=$R/exp9_debate_ppo_v6_stratdpo_base
CK="sd120=$RSD/checkpoint-0120 (merged: exp9-v6sd-tgt-c120)"
DEFS=${DEFS:-wmsr,cpwbft,sac}
if [ -n "${SMOKE:-}" ]; then N=2; SEEDS=3; TAG=def_smoke; else N=400; SEEDS=3,4,5; TAG=def; fi
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
COMMON="--victim $M/qwen3.5-9b --seeds $SEEDS --n $N --answer-first --profile-set $PS --tasks $TASKS --arg-tokens 200 --targets $TGT"

for DEF in ${DEFS//,/ }; do
  RUN=$R/${TAG}_${DEF}_none
  ts "START $DEF none -> $RUN"
  if [ ! -f $RUN/state.json ]; then
    python -m debate_ppo.eval_debate init --run $RUN --label "${TAG}_${DEF}_none_honest_suggestible_neutral_neutral" \
      --attackers '{}' --conds none --defense $DEF $COMMON
  fi
  python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1

  RUN=$R/${TAG}_${DEF}_sd120
  ts "START $DEF sd120 -> $RUN"
  if [ ! -f $RUN/state.json ]; then
    python -m debate_ppo.eval_debate init --run $RUN --label "${TAG}_${DEF}_sd120_honest_suggestible_neutral_neutral" \
      --attackers "{\"targeted_model_rl_sd120\":\"$M/exp9-v6sd-tgt-c120\"}" --conds targeted_model_rl_sd120 \
      --defense $DEF --rl-checkpoint "$CK" $COMMON
  fi
  python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1

  RUN=$R/${TAG}_${DEF}_dwmbon
  ts "START $DEF dwmbon -> $RUN"
  if [ ! -f $RUN/state.json ]; then
    python -m debate_ppo.eval_debate_dwm init --run $RUN --label "${TAG}_${DEF}_dwmbon_honest_suggestible_neutral_neutral" \
      --attackers "{\"targeted_model_rl_sd120__wmbon8dw\":\"$M/exp9-v6sd-tgt-c120\"}" --conds targeted_model_rl_sd120__wmbon8dw \
      --defense $DEF --rl-checkpoint "$CK" --gpu-util 0.6 --dwm-model $M/debate-wm-v1 $COMMON
  fi
  python -m debate_ppo.eval_debate_dwm play --run $RUN >> $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
  ts "DONE $DEF"
done
[ -n "${SMOKE:-}" ] || python debate_wm/def_paired.py --tag def > $R/def_filters_paired.md 2> $R/def_filters_paired.err || true
ts "DEF_FILTERS_DONE"
