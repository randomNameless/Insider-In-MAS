#!/usr/bin/env bash
# Literature baseline (2026-10-20): Amayuelas et al. 2024 prompt-based adversary, honest = real Qwen3.5-9B.
# Same split / seeds (3,4,5) / targets as exp9_v6_targeted_test_9b; none + targeted_4b re-run in-batch as references.
# Their Best-of-N judge arm is not part of this run.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
ATT=$(printf '{"targeted_4b":"%s","amayuelas_4b":"%s","amayuelas_9b":"%s"}' $M/qwen3-4b $M/qwen3-4b $M/qwen3.5-9b)
RUN=/path/to/runs/exp9_amayuelas_test_9b
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "amayuelas2024_prompt_adversary_test_9b" --victim $M/qwen3.5-9b \
    --attackers "$ATT" --conds none,targeted_4b,amayuelas_4b,amayuelas_9b --seeds 3,4,5 --n 400 \
    --answer-first --profile-set v1 --targets /path/to/runs/targets_none9b.json
fi
python -m debate_ppo.eval_debate play --run $RUN > $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo AMAYUELAS_DONE
