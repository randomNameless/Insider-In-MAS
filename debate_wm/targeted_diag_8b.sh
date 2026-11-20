#!/usr/bin/env bash
# Targeted diagnostic with honest = Qwen3-8B (2026-10-18). Evaluation of existing models only.
# Targets are recomputed from the 8B `none` run (its own round-1 wrong-option votes), so the assigned
# option is the one THESE honest agents find attractive. test 400 x seeds 0,1; same seats/honest seeds
# as exp9_eval_honest_qwen3_8b, so `none` / `prompt` are comparable with that run.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
T=/path/to/runs/targets_none8b.json
[ -f $T ] || python -m debate_wm.make_targets /path/to/runs/exp9_eval_honest_qwen3_8b/state.json --out $T
ATT=$(printf '{"prompt":"%s","targeted_4b":"%s","targeted_9b":"%s"}' $M/qwen3-4b $M/qwen3-4b $M/qwen3.5-9b)
RUN=/path/to/runs/exp9_targeted_diag_8b
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "targeted_diagnostic_honest_qwen3_8b" --victim $M/qwen3-8b \
    --attackers "$ATT" --conds none,prompt,targeted_4b,targeted_9b --seeds 0,1 --n 400 \
    --answer-first --profile-set v1 --targets $T
fi
python -m debate_ppo.eval_debate play --run $RUN > $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
python -m debate_wm.influence $RUN/state.json --out /path/to/runs/influence/targeted_8b.json > /dev/null
echo TARGETED8B_DONE
