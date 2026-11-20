#!/usr/bin/env bash
# Belief-WM Best-of-N at inference for the best targeted PPO persuader (2026-10-25). Same test set as every other batch
# (MMLU-Pro test n=400, seeds 3,4,5, targets_none9b.json, profile set v1, honest = Qwen3.5-9B, --answer-first --arg-tokens 200).
# Persuader = exp9-v6-tgt-c120 (targeted PPO v6 on the PLAIN Qwen3-4B base, step 120; single-sample 0.456-0.467). 2026-10-27: PPO-only + WM-BoN8.
#   targeted_model_rl_c120__wmbon8wb  8 candidates/turn, WM tracks beliefs itself (deployable)
#   targeted_model_rl_c120__wmbon8tb  8 candidates/turn, true honest readouts fed to the WM (upper-bound diagnostic)
# Baselines (none, single-sample sd120) are the paired debates of exp9_v6sd_ours_test_9b (same seats / honest seeds).
# gpu-util 0.6 leaves room for the WM judge next to the vLLM engine. SMOKE: RUN=~/runs/smoke_wmbon N=2 SEEDS=3 $0
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
ATT="{\"targeted_model_rl_c120__wmbon8wb\":\"$M/exp9-v6-tgt-c120\",\"targeted_model_rl_c120__wmbon8tb\":\"$M/exp9-v6-tgt-c120\"}"
RUN=${RUN:-/path/to/runs/exp9_v6_c120_wmbon_test_9b}
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "belief_wm_bon8_on_v6_c120_plainbase_test_9b_h100" --victim $M/qwen3.5-9b \
    --attackers "$ATT" --conds ${CONDS:-targeted_model_rl_c120__wmbon8wb,targeted_model_rl_c120__wmbon8tb} \
    --seeds ${SEEDS:-3,4,5} --n ${N:-400} --answer-first --profile-set v1 --arg-tokens 200 --gpu-util 0.6 \
    --targets /path/to/runs/targets_none9b.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl \
    --rl-checkpoint "c120=/path/to/runs/exp9_debate_ppo_v6_targeted/checkpoint-0120 (merged: exp9-v6-tgt-c120)"
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo WMBON_EVAL_DONE
