#!/usr/bin/env bash
# Defense thread (2026-11-08): collect attacker trajectories on the FIT split for adversarial training of the
# honest victims. Same harness/flags as exp10_nn (v1nn, Qwen3.5-9B honest, answer-first, 200 arg tokens), but
# split=fit (1441 q), seed 0, so training prompts never overlap the test 400. Clean reference = the existing
# stratdpo6_collect_v1nn_fit `none` run (same split/seed/profile set); targets = its targets.json.
# Runs: def_fit_sd120 (targeted_model_rl_sd120) -> def_fit_dwmbon (targeted_model_rl_sd120__wmbon8dw).
# setsid nohup bash ~/consensus-attacker/debate_wm/def_collect_fit_lambda.sh > ~/runs/def_collect_fit.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
M=/path/to/models; R=/path/to/runs; PS=v1nn
TASKS=/path/to/data/mmlupro_v1/tasks.jsonl
TGT=$R/stratdpo6_collect_v1nn_fit/targets.json
RSD=$R/exp9_debate_ppo_v6_stratdpo_base
CK="sd120=$RSD/checkpoint-0120 (merged: exp9-v6sd-tgt-c120)"
N=${N:-1441}; SEEDS=${SEEDS:-0}; TAG=${TAG:-def_fit}
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

RUN=$R/${TAG}_sd120
ts "START sd120 -> $RUN"
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "${TAG}_sd120_honest_suggestible_neutral_neutral" \
    --split fit --victim $M/qwen3.5-9b --attackers "{\"targeted_model_rl_sd120\":\"$M/exp9-v6sd-tgt-c120\"}" \
    --conds targeted_model_rl_sd120 --seeds $SEEDS --n $N --answer-first --profile-set $PS --tasks $TASKS \
    --arg-tokens 200 --targets $TGT --rl-checkpoint "$CK"
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
ts "DONE sd120"

RUN=$R/${TAG}_dwmbon
ts "START dwmbon -> $RUN"
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate_dwm init --run $RUN --label "${TAG}_dwmbon_honest_suggestible_neutral_neutral" \
    --split fit --victim $M/qwen3.5-9b --attackers "{\"targeted_model_rl_sd120__wmbon8dw\":\"$M/exp9-v6sd-tgt-c120\"}" \
    --conds targeted_model_rl_sd120__wmbon8dw --seeds $SEEDS --n $N --answer-first --profile-set $PS --tasks $TASKS \
    --arg-tokens 200 --gpu-util 0.6 --targets $TGT --rl-checkpoint "$CK" --dwm-model $M/debate-wm-v1
fi
python -m debate_ppo.eval_debate_dwm play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
ts "DONE dwmbon"
ts "DEF_COLLECT_FIT_DONE"
