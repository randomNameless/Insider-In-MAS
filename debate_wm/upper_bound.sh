#!/usr/bin/env bash
# Upper bound of attack effect under the debate protocol (honest = real Qwen3.5-9B, v1 personas):
# none / model_base (reference) / prompt (4B, v3 prompt) / prompt_9b (9B attacker, v3 prompt, no gold) /
# gold_4b (Qwen3-4B told the correct answer) / gold_9b (Qwen3.5-9B told the correct answer).
# Test 400 x seeds 0,1 (same seats / honest seeds as the deliverables).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
ATT=$(printf '{"model_base":"%s","prompt":"%s","prompt_9b":"%s","gold_4b":"%s","gold_9b":"%s"}' \
  $M/qwen3-4b $M/qwen3-4b $M/qwen3.5-9b $M/qwen3-4b $M/qwen3.5-9b)
RUN=/path/to/runs/exp9_upper_bound_9b
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "upper_bound_honest_9b" --victim $M/qwen3.5-9b \
    --attackers "$ATT" --conds none,model_base,prompt,prompt_9b,gold_4b,gold_9b --seeds 0,1 --n 400 \
    --answer-first --profile-set v1
fi
python -m debate_ppo.eval_debate play --run $RUN > $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo UPPER_DONE
