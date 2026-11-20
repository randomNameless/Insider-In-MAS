#!/usr/bin/env bash
# exp10_nn batch for the v6-format strat-DPO (2026-10-28; plan ~/runs/ppo_diag_20261028/STRATDPO_V6_PLAN.md).
# Same harness as eval_nn_chain_lambda.sh: honest = Qwen3.5-9B, profile v1nn (suggestible + neutral + neutral), MMLU-Pro
# test n=400, seeds 3,4,5, targets_none9b_exp10_nn.json, v6 decoder (--answer-first, target prefilled, T 1.0, 200 tokens).
#   targeted_model_stratdpo6  new DPO model (debate_ppo/strat_dpo/run_v6.sh)
#   targeted_model_stratdpo   old strat-DPO weights in the v6 format = the PPO step-0 control for sd120
# Paired against exp10_nn_none (same task/seed/seat/honest seeds) by debate_wm/nn_paired_dpo6.py.
# SMOKE: RUN=~/runs/smoke_nn_dpo6 N=2 SEEDS=3 MODEL6=~/models/strat-dpo-v6-q3-4b-smoke $0
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models; R=/path/to/runs
MODEL6=${MODEL6:-$M/strat-dpo-v6-q3-4b}
RUN=${RUN:-$R/exp10_nn_dpo6}
ATT=$(printf '{"targeted_model_stratdpo6":"%s","targeted_model_stratdpo":"%s"}' $MODEL6 $M/strat-dpo-q3-4b)
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "exp10_nn_dpo6_honest_suggestible_neutral_neutral" \
    --victim $M/qwen3.5-9b --attackers "$ATT" --conds targeted_model_stratdpo6,targeted_model_stratdpo \
    --seeds ${SEEDS:-3,4,5} --n ${N:-400} --answer-first --profile-set v1nn --arg-tokens 200 \
    --targets $R/targets_none9b_exp10_nn.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl
  for m in $MODEL6 $M/strat-dpo-q3-4b $M/qwen3.5-9b; do
    ( cd $m && sha256sum *.safetensors | sed "s#  #  $(basename $m)/#" )
  done > $RUN/models_sha256.txt
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo DPO6_EVAL_DONE
