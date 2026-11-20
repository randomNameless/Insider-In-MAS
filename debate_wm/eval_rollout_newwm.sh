#!/usr/bin/env bash
# Rollout-level check of the distilled debate WM: identical setup to Exp9 deliverables 1/2
# (test 400 x seeds 0,1,2, 7 attacker conditions, same seats / honest seeds), honest agents = new WM.
# Compare with ~/runs_lambda .../exp9_eval_v4b/{wm,9b}/results.json (old WM / real 9B).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
until [ -f $M/exp9-v4b-rl-last/config.json ] && ! pgrep -f merge_adapter >/dev/null; do sleep 30; done
ATT=$(printf '{"prompt":"%s","model_base":"%s","model_base_argfirst":"%s","model_rl_ckpt20":"%s","model_rl_best":"%s","model_rl_last":"%s"}' \
  $M/qwen3-4b $M/qwen3-4b $M/qwen3-4b $M/exp9-v4b-rl-ckpt20 $M/exp9-v4b-rl-best $M/exp9-v4b-rl-last)
RUN=/path/to/runs/exp9_eval_newwm
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "rollout_eval_debate_wm_v1" --victim $M/debate-wm-v1 \
    --attackers "$ATT" --rl-checkpoint "ckpt20=v4b/checkpoint-0020 best=v4b/best(step140) last=v4b/interrupted-0172" \
    --conds none,prompt,model_base,model_base_argfirst,model_rl_ckpt20,model_rl_best,model_rl_last \
    --seeds 0,1,2 --n 400 --answer-first --profile-set v1
fi
python -m debate_ppo.eval_debate play --run $RUN > $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo ROLLOUT_DONE
