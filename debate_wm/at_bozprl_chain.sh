#!/usr/bin/env bash
# Bozdag P-RL (no private <think>, <=200 shown tokens) vs AT-DPO honest agents (2026-11-13).
# Same flags as eval_nn_bozdag_prl_lambda.sh (single) / bozprl_wm_chain.sh (+WM-BoN8 peek), but --victim at-dpo-9b and
# --arg-tokens 200 as in at_dpo_chain_lambda.sh. Runs: ~/runs/at_bozprl_{single,dwmbon}; SMOKE=1 -> at_smoke_bozprl_*.
# setsid nohup bash debate_wm/at_bozprl_chain.sh > ~/runs/at_bozprl_chain.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
M=/path/to/models; R=/path/to/runs
if [ -n "${SMOKE:-}" ]; then SEEDS=3; N=2; P=at_smoke_bozprl; else SEEDS=3,4,5; N=400; P=at_bozprl; fi
COMMON="--victim $M/at-dpo-9b --seeds $SEEDS --n $N --answer-first --profile-set v1nn --arg-tokens 200 --max-model-len 16384
  --targets $R/targets_none9b_exp10_nn.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl"
ts() { echo "[$(date -u +%H:%M:%S)] $*"; }
C=bozdag_prl_3b_nothink_len200; RUN=$R/${P}_single
[ -f $RUN/state.json ] || { python -m debate_ppo.eval_debate init --run $RUN --label "$(basename $RUN)_honest_suggestible_neutral_neutral" \
  --attackers "{\"$C\":\"$M/bozdag-prl-q25-3b\"}" --conds $C $COMMON
  for m in $M/bozdag-prl-q25-3b $M/at-dpo-9b; do ( cd $m && sha256sum *.safetensors | sed "s#  #  $(basename $m)/#" ); done > $RUN/models_sha256.txt; }
ts "START $RUN"
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
ts "SINGLE_DONE"
C=bozdag_prl_3b_nothink_len200_wmbon8dw; RUN=$R/${P}_dwmbon
[ -f $RUN/state.json ] || python -m debate_ppo.eval_debate_dwm_lit init --run $RUN --label "$(basename $RUN)_honest_suggestible_neutral_neutral" \
  --attackers "{\"$C\":\"$M/bozdag-prl-q25-3b\"}" --conds $C $COMMON --gpu-util 0.6 --dwm-model $M/debate-wm-v1
ts "START $RUN"
python -m debate_ppo.eval_debate_dwm_lit play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
ts "AT_BOZPRL_DONE"
