#!/usr/bin/env bash
# Real-9B test of the targeted PPO persuader (v6 / v6b, 2026-10-19). Checkpoints fixed IN ADVANCE, no selection
# on test: v6 checkpoint-0040 (switch point), v6b checkpoint-0080 (best WM val), v6b checkpoint-0120 (last).
# Targets = targets_none9b.json (from the 9B `none` run of exp9_targeted_diag, seeds 0,1); evaluated on seeds 3,4,5 so
# the honest round-1 samples that chose the targets are not the ones being attacked.
# targeted_model_base = untrained Qwen3-4B, same hint, letter prefilled, same decoder = PPO step 0.
# targeted_4b = untrained Qwen3-4B with the hint, free-form (the 10-18 diagnostic condition).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
R6=/path/to/runs/exp9_debate_ppo_v6_targeted
R6B=/path/to/runs/exp9_debate_ppo_v6b_targeted_len200
declare -A CK=([c040]=$R6/checkpoint-0040 [c080]=$R6B/checkpoint-0080 [c120]=$R6B/checkpoint-0120)
for k in c040 c080 c120; do
  OUT=$M/exp9-v6-tgt-$k
  [ -f $OUT/config.json ] || python scripts/merge_adapter.py --base $M/qwen3-4b --adapter ${CK[$k]} --out $OUT --dtype float32
done
ATT=$(printf '{"targeted_4b":"%s","targeted_model_base":"%s","targeted_model_rl_c040":"%s","targeted_model_rl_c080":"%s","targeted_model_rl_c120":"%s"}' \
  $M/qwen3-4b $M/qwen3-4b $M/exp9-v6-tgt-c040 $M/exp9-v6-tgt-c080 $M/exp9-v6-tgt-c120)
RUN=/path/to/runs/exp9_v6_targeted_test_9b
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "v6_targeted_ppo_test_9b" --victim $M/qwen3.5-9b \
    --attackers "$ATT" --conds none,targeted_4b,targeted_model_base,targeted_model_rl_c040,targeted_model_rl_c080,targeted_model_rl_c120 \
    --seeds 3,4,5 --n 400 --answer-first --profile-set v1 --arg-tokens 200 \
    --targets /path/to/runs/targets_none9b.json \
    --rl-checkpoint "c040=$R6/checkpoint-0040 c080=$R6B/checkpoint-0080 c120=$R6B/checkpoint-0120"
fi
python -m debate_ppo.eval_debate play --run $RUN > $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
python -m debate_wm.influence $RUN/state.json --out /path/to/runs/influence/v6_targeted_test_9b.json > /dev/null
echo V6_EVAL_DONE
