#!/usr/bin/env bash
# Strategy-catalog DPO model evaluated in the v6 PPO attacker format (2026-10-24). Evaluation only, no training.
# v6 format = eval_v6_ours_lambda.sh: malicious_v3 system prompt + TARGET_HINT, --answer-first with the target letter
# PREFILLED, argument sampled at T=1.0 until newline / 200 tokens. Same questions as every other batch: MMLU-Pro test
# n=400, seeds 3,4,5, targets_none9b.json; honest = Qwen3.5-9B.
#   none + targeted_model_stratdpo (strat-dpo-q3-4b, trained on the catalog prompt, never on this format) only
# (2026-10-24): targeted_model_base and v6 c040 already ran on the same test set (Lambda 10-19 exp9 v6 targeted
# test; c040 also ~/runs/exp9_v6_ours_test_9b 10-22), so they are compared across batches, each vs its own none.
# ANCHORS=1 re-runs them in this batch.
# Smoke: RUN=~/runs/smoke_stratdpo_v6fmt N=5 SEEDS=3 ANCHORS=1 $0
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
CONDS=none,targeted_model_stratdpo
ATT=$(printf '{"targeted_model_stratdpo":"%s"}' $M/strat-dpo-q3-4b)
if [ -n "${ANCHORS:-}" ]; then
  CONDS=none,targeted_model_base,targeted_model_stratdpo,targeted_model_rl_c040
  ATT=$(printf '{"targeted_model_base":"%s","targeted_model_stratdpo":"%s","targeted_model_rl_c040":"%s"}' \
    $M/qwen3-4b $M/strat-dpo-q3-4b $M/exp9-v6-tgt-c040)
fi
RUN=${RUN:-/path/to/runs/exp9_stratdpo_v6fmt_test_9b}
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "stratdpo_in_v6_format_test_9b_h100" --victim $M/qwen3.5-9b \
    --attackers "$ATT" --conds $CONDS \
    --seeds ${SEEDS:-3,4,5} --n ${N:-400} --answer-first --profile-set v1 --arg-tokens 200 \
    --targets /path/to/runs/targets_none9b.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl
  for m in qwen3-4b strat-dpo-q3-4b exp9-v6-tgt-c040 qwen3.5-9b; do
    ( cd $M/$m && sha256sum *.safetensors | sed "s#  #  $m/#" )
  done > $RUN/models_sha256.txt
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo STRATDPO_V6FMT_DONE
