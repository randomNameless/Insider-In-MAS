#!/usr/bin/env bash
# Targeted diagnostic (2026-10-18): does ASSIGNING one specific wrong option make a prompt-only
# attacker more effective? Evaluation of existing models only; no training.
# honest = real Qwen3.5-9B (v1 personas), protocol unchanged, test 400 x seeds 0,1 (same seats /
# honest seeds as exp9_upper_bound_9b, so `none` and `prompt` are directly comparable).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
T=/path/to/runs/targets_none9b.json
[ -f $T ] || python -m debate_wm.make_targets /path/to/runs/exp9_upper_bound_9b/state.json --out $T
ATT=$(printf '{"prompt":"%s","targeted_4b":"%s","targeted_9b":"%s"}' $M/qwen3-4b $M/qwen3-4b $M/qwen3.5-9b)
RUN=/path/to/runs/exp9_targeted_diag
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "targeted_diagnostic_honest_9b" --victim $M/qwen3.5-9b \
    --attackers "$ATT" --conds none,prompt,targeted_4b,targeted_9b --seeds 0,1 --n 400 \
    --answer-first --profile-set v1 --targets $T
fi
python -m debate_ppo.eval_debate play --run $RUN > $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo TARGETED_DONE
