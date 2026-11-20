#!/usr/bin/env bash
# RQ4 ablation (2026-11-05; plan doc "RQ4 Ablation: inventory and plan"). exp10_nn harness, flags as dwm_chain_lambda.sh /
# eval_nn_chain_lambda.sh; paired against exp10_nn_none by debate_wm/abl_paired.py.
#   run abl_dwm:     targeted_model_base__wmbon8dw (pi_ref + dwm-BoN8), targeted_model_stratdpo__wmbon8dw (DPO step 0 + dwm-BoN8)
#   run abl_rlcurve: PPO checkpoints 040/080 of c (from pi_ref) and sd (from strat-DPO), single sample
# setsid nohup bash ~/consensus-attacker/debate_wm/abl_chain_lambda.sh > ~/runs/abl_chain.log 2>&1 &
# SMOKE=1: N=2 SEEDS=3, *_smoke dirs.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
M=/path/to/models; R=/path/to/runs; TAG=exp10_nn
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
S=${SMOKE:+_smoke}
if [ -n "$S" ]; then SEEDS=3; N=2; else SEEDS=3,4,5; N=400; fi
RSD=$R/exp9_debate_ppo_v6_stratdpo_base; R6B=$R/exp9_debate_ppo_v6b_targeted_len200
COMMON="--victim $M/qwen3.5-9b --seeds $SEEDS --n $N --answer-first --profile-set v1nn --tasks /path/to/data/mmlupro_v1/tasks.jsonl --arg-tokens 200 --targets $R/targets_none9b_${TAG}.json"

RUN=$R/${TAG}_abl_dwm$S
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate_dwm init --run $RUN --label "${TAG}_abl_dwm_honest_suggestible_neutral_neutral" \
    --attackers "{\"targeted_model_base__wmbon8dw\":\"$M/qwen3-4b\",\"targeted_model_stratdpo__wmbon8dw\":\"$M/strat-dpo-q3-4b\"}" \
    --conds targeted_model_base__wmbon8dw,targeted_model_stratdpo__wmbon8dw $COMMON --gpu-util 0.6 --dwm-model $M/debate-wm-v1
fi
ts "START abl_dwm -> $RUN"
python -m debate_ppo.eval_debate_dwm play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
ts "DONE abl_dwm"

RUN=$R/${TAG}_abl_rlcurve$S
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "${TAG}_abl_rlcurve_honest_suggestible_neutral_neutral" \
    --attackers "{\"targeted_model_rl_c040\":\"$M/exp9-v6-tgt-c040\",\"targeted_model_rl_c080\":\"$M/exp9-v6-tgt-c080\",\"targeted_model_rl_sd040\":\"$M/exp9-v6sd-tgt-c040\",\"targeted_model_rl_sd080\":\"$M/exp9-v6sd-tgt-c080\"}" \
    --conds targeted_model_rl_c040,targeted_model_rl_c080,targeted_model_rl_sd040,targeted_model_rl_sd080 $COMMON \
    --rl-checkpoint "c040/c080=$R6B/checkpoint-00{40,80} (merged: exp9-v6-tgt-c0{40,80}) sd040/sd080=$RSD/checkpoint-00{40,80} (merged: exp9-v6sd-tgt-c0{40,80})"
fi
ts "START abl_rlcurve -> $RUN"
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
ts "DONE abl_rlcurve"
[ -n "$S" ] || python debate_wm/abl_paired.py > $R/${TAG}_abl_paired.md 2> $R/${TAG}_abl_paired.err
ts "ABL_CHAIN_DONE"
