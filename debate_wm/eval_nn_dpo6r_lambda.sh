#!/usr/bin/env bash
# exp10_nn batch for the regularised v6-format strat-DPO (2026-10-30). Same harness as eval_nn_dpo6_lambda.sh
# (Qwen3.5-9B honest, v1nn, test n=400, seeds 3,4,5, targets_none9b_exp10_nn.json, v6 decoder) but --arg-tokens 500
# (token limit raised to 500). Conditions (names distinct from the 200-token batch so nn_paired can hold both):
#   targeted_model_stratdpo6r   new regularised model (debate_ppo/strat_dpo/run_v6r.sh)
#   targeted_model_stratdpo500  old strat-DPO weights, v6 format, 500-token cap = the token-cap control
# SMOKE: RUN=~/runs/smoke_nn_dpo6r N=2 SEEDS=3 MODEL6R=~/models/strat-dpo-v6r-q3-4b-smoke $0
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models; R=/path/to/runs
MODEL6R=${MODEL6R:-$M/strat-dpo-v6r-q3-4b}
RUN=${RUN:-$R/exp10_nn_dpo6r}
ATT=$(printf "{\"targeted_model_stratdpo6r\":\"%s\",\"targeted_model_stratdpo500\":\"%s\"}" $MODEL6R $M/strat-dpo-q3-4b)
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "exp10_nn_dpo6r_honest_suggestible_neutral_neutral_argtok500" \
    --victim $M/qwen3.5-9b --attackers "$ATT" --conds targeted_model_stratdpo6r,targeted_model_stratdpo500 \
    --seeds ${SEEDS:-3,4,5} --n ${N:-400} --answer-first --profile-set v1nn --arg-tokens ${ARGTOK:-500} \
    --targets $R/targets_none9b_exp10_nn.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl
  for m in $MODEL6R $M/strat-dpo-q3-4b $M/qwen3.5-9b; do
    ( cd $m && sha256sum *.safetensors | sed "s#  #  $(basename $m)/#" )
  done > $RUN/models_sha256.txt
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo DPO6R_EVAL_DONE
