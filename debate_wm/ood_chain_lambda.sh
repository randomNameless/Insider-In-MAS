#!/usr/bin/env bash
# 2026-11-02: zero-shot OOD transfer of the MMLU-Pro-trained insider. Same harness as eval_nn_chain_lambda.sh
# (v1nn, Qwen3.5-9B honest, seeds 3,4,5, --answer-first, arg-tokens 200) + dwm_chain_lambda.sh (debate-wm-v1 BoN8),
# on a different tasks file. Targets from the dataset's own `none` round 1. Conds: none -> ours (sd120, base, 4b)
# -> dwm (sd120__wmbon8dw, base__wmbon8dw = WM selection without the RL policy).
# DS=truthfulqa_mc1 setsid nohup bash ~/consensus-attacker/debate_wm/ood_chain_lambda.sh > ~/runs/ood_truthfulqa_mc1_chain.log 2>&1 &
# Smoke: DS=truthfulqa_mc1 TAG=ood_smoke N=2 SEEDS=3 $0
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
M=/path/to/models; R=/path/to/runs; PS=v1nn
DS=${DS:?set DS}; TAG=${TAG:-ood_$DS}; N=${N:-400}; SEEDS=${SEEDS:-3,4,5}
TASKS=/path/to/data/ood_$DS/tasks.jsonl
TGT=$R/targets_none9b_${TAG}.json
RSD=$R/exp9_debate_ppo_v6_stratdpo_base
CKPT="sd120=$RSD/checkpoint-0120 (merged: exp9-v6sd-tgt-c120)"
COMMON="--victim $M/qwen3.5-9b --seeds $SEEDS --n $N --answer-first --profile-set $PS --tasks $TASKS"
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
run() {  # name module conds attackers [extra init args]
  local name=$1 mod=$2 conds=$3 att=$4; shift 4
  local RUN=$R/${TAG}_$name
  ts "START $name conds=$conds"
  [ -f $RUN/state.json ] || python -m $mod init --run $RUN --label "${TAG}_${name}_honest_suggestible_neutral_neutral" \
      --attackers "$att" --conds $conds $COMMON "$@"
  python -m $mod play --run $RUN >> $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
  ts "DONE $name"
}
run none debate_ppo.eval_debate none '{}'
[ -f $TGT ] || python -m debate_wm.make_targets $R/${TAG}_none/state.json --out $TGT
run ours debate_ppo.eval_debate targeted_model_rl_sd120,targeted_model_base,targeted_4b \
  "{\"targeted_model_rl_sd120\":\"$M/exp9-v6sd-tgt-c120\",\"targeted_model_base\":\"$M/qwen3-4b\",\"targeted_4b\":\"$M/qwen3-4b\"}" \
  --arg-tokens 200 --targets $TGT --rl-checkpoint "$CKPT"
run dwmbon debate_ppo.eval_debate_dwm targeted_model_rl_sd120__wmbon8dw,targeted_model_base__wmbon8dw \
  "{\"targeted_model_rl_sd120__wmbon8dw\":\"$M/exp9-v6sd-tgt-c120\",\"targeted_model_base__wmbon8dw\":\"$M/qwen3-4b\"}" \
  --arg-tokens 200 --gpu-util 0.6 --targets $TGT --rl-checkpoint "$CKPT" --dwm-model $M/debate-wm-v1
ts OOD_CHAIN_DONE
