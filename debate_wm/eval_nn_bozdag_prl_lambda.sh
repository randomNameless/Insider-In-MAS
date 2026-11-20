#!/usr/bin/env bash
# Bozdag et al. 2026 RELEASED P-RL persuader (2026-11-12): beyzabozdag/p-rl_qwen2-5-3b (full GRPO fine-tune of
# Qwen2.5-3B-Instruct, gated on HF) as the Bozdag literature row of the exp10_nn table. Replaces bozdag_q25_7b in
# exp10_nn_lit (untrained Qwen2.5-7B-Instruct + their static prompt); our own GRPO repro weights were deleted 11-12.
# Same harness as the lit batch of eval_nn_chain_lambda.sh: Qwen3.5-9B honest, v1nn, MMLU-Pro test n=400, seeds 3,4,5,
# targets_none9b_exp10_nn.json, --max-model-len 16384, their persuader prompt + decoding (baseline_bozdag.py, T=0.7,
# 2048 tokens). Run dir matches exp10_nn_* so debate_wm/nn_paired.py exp10_nn pairs it against exp10_nn_none.
# Word-budget control (11-12): COND=bozdag_prl_3b_w100 RUN=~/runs/exp10_nn_bozprl_w100 $0 (see baseline_bozdag.WORD_RULE).
# Our own budget (11-12): COND=bozdag_prl_3b_len200 RUN=~/runs/exp10_nn_bozprl_len200 $0 (200 tokens = our --arg-tokens).
# SMOKE: RUN=~/runs/smoke_nn_bozprl N=2 SEEDS=3 $0   (outside the exp10_nn_* glob)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models; R=/path/to/runs
MODEL=${MODEL:-$M/bozdag-prl-q25-3b}
RUN=${RUN:-$R/exp10_nn_bozprl}; COND=${COND:-bozdag_prl_3b}
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "$(basename $RUN)_honest_suggestible_neutral_neutral" \
    --victim $M/qwen3.5-9b --attackers "{\"$COND\":\"$MODEL\"}" --conds $COND \
    --seeds ${SEEDS:-3,4,5} --n ${N:-400} --answer-first --profile-set v1nn --max-model-len 16384 \
    --targets $R/targets_none9b_exp10_nn.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl
  for m in $MODEL $M/qwen3.5-9b; do
    ( cd $m && sha256sum *.safetensors | sed "s#  #  $(basename $m)/#" )
  done > $RUN/models_sha256.txt
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo BOZPRL_EVAL_DONE
