#!/usr/bin/env bash
# Re-validation with a different honest model (2026-10-17): honest agents = Qwen3-8B (v1 personas),
# everything else identical to the Exp9 deliverables (test 400 x seeds 0,1,2, same seats / honest seeds).
# Attackers = existing models only (no new training).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
ATT=$(printf '{"prompt":"%s","model_base":"%s","model_base_argfirst":"%s","model_rl_oldwm_c20":"%s","model_rl_oldwm_best":"%s","model_rl_oldwm_last":"%s","model_rl_newwm_c120":"%s","prompt_9b":"%s"}' \
  $M/qwen3-4b $M/qwen3-4b $M/qwen3-4b $M/exp9-v4b-rl-ckpt20 $M/exp9-v4b-rl-best $M/exp9-v4b-rl-last $M/exp9-v5-rl-c0120 $M/qwen3.5-9b)
RUN=/path/to/runs/exp9_eval_honest_qwen3_8b
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "revalidate_honest_qwen3_8b" --victim $M/qwen3-8b \
    --attackers "$ATT" --conds none,prompt,model_base,model_base_argfirst,model_rl_oldwm_c20,model_rl_oldwm_best,model_rl_oldwm_last,model_rl_newwm_c120,prompt_9b \
    --seeds 0,1,2 --n 400 --answer-first --profile-set v1 \
    --rl-checkpoint "oldwm: v4b ckpt20/best(step140)/last(step172); newwm: v5 c0120"
fi
python -m debate_ppo.eval_debate play --run $RUN > $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo EVAL8B_DONE
