#!/usr/bin/env bash
# Literature baselines (2026-10-20): Bozdag et al. 2026 static-prompt persuader (debate_ppo/baseline_bozdag.py) and
# Kraidia et al. 2026 prompt-only adversary PPA (debate_ppo/baseline_kraidia.py), honest = real Qwen3.5-9B.
# Same split / seeds (3,4,5) / targets as exp9_v6_targeted_test_9b; none + targeted_4b re-run in-batch as references.
# Not part of this run: Bozdag GRPO persuaders, Kraidia Best-of-N and RAG arms.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
ATT=$(printf '{"targeted_4b":"%s","bozdag_4b":"%s","bozdag_9b":"%s","kraidia_4b":"%s","kraidia_9b":"%s"}' $M/qwen3-4b $M/qwen3-4b $M/qwen3.5-9b $M/qwen3-4b $M/qwen3.5-9b)
RUN=/path/to/runs/exp9_bozdag_kraidia_test_9b
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "bozdag2026_kraidia2026_prompt_adversaries_test_9b" --victim $M/qwen3.5-9b \
    --attackers "$ATT" --conds none,targeted_4b,bozdag_4b,bozdag_9b,kraidia_4b,kraidia_9b --seeds 3,4,5 --n 400 \
    --answer-first --profile-set v1 --targets /path/to/runs/targets_none9b.json
fi
python -m debate_ppo.eval_debate play --run $RUN > $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo BOZDAG_KRAIDIA_DONE
