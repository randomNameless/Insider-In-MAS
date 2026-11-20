#!/usr/bin/env bash
# Our targeted PPO persuader only (2026-10-22), Lambda H100, honest = real Qwen3.5-9B. Same questions as the
# literature-baseline runs (runs_local/baselines_test_9b, runs_local/tomap_test_9b): MMLU-Pro test n=400, seeds 3,4,5,
# targets_none9b.json. Our decoding as debate_wm/eval_v6_targeted.sh: --answer-first (target letter prefilled),
# --arg-tokens 200, checkpoints fixed in advance. `none` re-run in-batch so "vs none" is paired within this run.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
R6=/path/to/runs/exp9_debate_ppo_v6_targeted
R6B=/path/to/runs/exp9_debate_ppo_v6b_targeted_len200
ATT=$(printf '{"targeted_model_rl_c040":"%s","targeted_model_rl_c080":"%s","targeted_model_rl_c120":"%s"}' \
  $M/exp9-v6-tgt-c040 $M/exp9-v6-tgt-c080 $M/exp9-v6-tgt-c120)
RUN=/path/to/runs/exp9_v6_ours_test_9b
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "v6_targeted_ppo_ours_test_9b_h100" --victim $M/qwen3.5-9b \
    --attackers "$ATT" --conds none,targeted_model_rl_c040,targeted_model_rl_c080,targeted_model_rl_c120 \
    --seeds 3,4,5 --n 400 --answer-first --profile-set v1 --arg-tokens 200 \
    --targets /path/to/runs/targets_none9b.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl \
    --rl-checkpoint "c040=$R6/checkpoint-0040 c080=$R6B/checkpoint-0080 c120=$R6B/checkpoint-0120"
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo V6_OURS_DONE
