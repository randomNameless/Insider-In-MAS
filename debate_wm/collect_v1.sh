#!/usr/bin/env bash
# Teacher trajectories for the debate WM: Qwen3.5-9B honest agents (profile set v1: stubborn /
# suggestible / neutral), 4th agent in {neutral honest, prompt attacker, model_base, PPO step-20},
# fit split (SFT train) then val split (SFT validation). Test split is NOT collected here.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
ATT=$(printf '{"prompt":"%s","model_base":"%s","model_rl_ckpt20":"%s"}' $M/qwen3-4b $M/qwen3-4b $M/exp9-v4b-rl-ckpt20)
for SPLIT in fit val; do
  if [ $SPLIT = fit ]; then N=1441; SEEDS=0,1; else N=160; SEEDS=0,1; fi
  RUN=/path/to/runs/wm_collect_v1_$SPLIT
  if [ ! -f $RUN/state.json ]; then
    python -m debate_ppo.eval_debate init --run $RUN --label "wm_teacher_9b_v1_$SPLIT" --split $SPLIT --n $N \
      --victim $M/qwen3.5-9b --attackers "$ATT" --conds none,prompt,model_base,model_rl_ckpt20 \
      --seeds $SEEDS --answer-first --profile-set v1
  fi
  python -m debate_ppo.eval_debate play --run $RUN > $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
  python -m debate_wm.build_sft --states $RUN/state.json --out /path/to/data/wm/v1_$SPLIT.jsonl
done
echo COLLECT_DONE
