#!/usr/bin/env bash
# 2026-10-31: (1) rank_oracle_exp10 re-check with the wm_judge_eval rankers (debate_wm/rank_oracle_judges.py);
# (2) end-to-end WM-BoN8 with debate-wm-v1 direct readout as selector (debate_ppo/eval_debate_dwm.py) in the exp10_nn harness:
#     same flags as `run wmbon` of eval_nn_chain_lambda.sh, conds *__wmbon8dw, paired against exp10_nn_{none,ours,wmbon}.
# setsid nohup bash ~/consensus-attacker/debate_wm/dwm_chain_lambda.sh > ~/runs/dwm_chain.log 2>&1 &
# SMOKE=1: step 1 on 96 listener turns, step 2 on N=2 SEEDS=3, separate dirs.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
M=/path/to/models; R=/path/to/runs; TAG=exp10_nn
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
S=${SMOKE:+_smoke}
if [ -n "$S" ]; then SEEDS=3; N=2; else SEEDS=3,4,5; N=400; fi
RES=$R/rank_oracle_exp10_judges$S
RJ="python debate_wm/rank_oracle_judges.py --out $R/rank_oracle_exp10 --res $RES ${SMOKE:+--limit 96}"
ts "step1 -> $RES"
$RJ --model $M/debate-wm-v1 --tag wm --judges roleplay_direct,roleplay_gen
$RJ --model $M/qwen3.5-4b --tag q4b --judges observer_v2
$RJ --model $M/qwen3.5-9b --tag q9b --judges observer_v2,roleplay_direct,roleplay_gen
[ -n "$S" ] || CUDA_VISIBLE_DEVICES= $RJ --analyze
ts "STEP1_DONE"
RUN=$R/${TAG}_dwmbon$S
R6B=$R/exp9_debate_ppo_v6b_targeted_len200; RSD=$R/exp9_debate_ppo_v6_stratdpo_base
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate_dwm init --run $RUN --label "${TAG}_dwmbon_honest_suggestible_neutral_neutral" \
    --victim $M/qwen3.5-9b --attackers "{\"targeted_model_rl_sd120__wmbon8dw\":\"$M/exp9-v6sd-tgt-c120\",\"targeted_model_rl_c120__wmbon8dw\":\"$M/exp9-v6-tgt-c120\"}" \
    --conds targeted_model_rl_sd120__wmbon8dw,targeted_model_rl_c120__wmbon8dw --seeds $SEEDS --n $N \
    --answer-first --profile-set v1nn --tasks /path/to/data/mmlupro_v1/tasks.jsonl \
    --arg-tokens 200 --gpu-util 0.6 --targets $R/targets_none9b_${TAG}.json \
    --rl-checkpoint "sd120=$RSD/checkpoint-0120 (merged: exp9-v6sd-tgt-c120) c120=$R6B/checkpoint-0120 (merged: exp9-v6-tgt-c120)" \
    --dwm-model $M/debate-wm-v1
fi
ts "step2 play -> $RUN"
python -m debate_ppo.eval_debate_dwm play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
[ -n "$S" ] || python debate_wm/nn_paired_dwm.py > $R/${TAG}_dwmbon_paired.md 2> $R/${TAG}_dwmbon_paired.err
ts "DWM_CHAIN_DONE"
