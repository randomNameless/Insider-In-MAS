#!/usr/bin/env bash
# Length-matched comparison (2026-10-22), Lambda H100, honest = real Qwen3.5-9B. Is kraidia_9b's strength its length
# (~400-word arguments vs ours <= 200 tokens)? kraidia_9b_len200 = kraidia_9b whose system prompt explicitly allows at most 200
# tokens per argument (eval_debate.LENGTH_RULE); over-length replies are shown up to their last complete sentence within 200
# tokens, never cut mid-sentence (eval_debate.fit_sentences). Ours (c040, c120) and full kraidia_9b in the same batch, so every
# comparison is paired. Same questions as all baseline runs: MMLU-Pro test n=400, seeds 3,4,5, targets_none9b.json.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
R6=/path/to/runs/exp9_debate_ppo_v6_targeted
R6B=/path/to/runs/exp9_debate_ppo_v6b_targeted_len200
ATT=$(printf '{"kraidia_9b":"%s","kraidia_9b_len200":"%s","targeted_model_rl_c040":"%s","targeted_model_rl_c120":"%s"}' \
  $M/qwen3.5-9b $M/qwen3.5-9b $M/exp9-v6-tgt-c040 $M/exp9-v6-tgt-c120)
RUN=/path/to/runs/exp9_lengthmatch_test_9b
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "length_matched_kraidia9b_vs_v6_test_9b_h100" --victim $M/qwen3.5-9b \
    --attackers "$ATT" --conds none,kraidia_9b,kraidia_9b_len200,targeted_model_rl_c040,targeted_model_rl_c120 \
    --seeds 3,4,5 --n 400 --answer-first --profile-set v1 --arg-tokens 200 --max-model-len 16384 \
    --targets /path/to/runs/targets_none9b.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl \
    --rl-checkpoint "c040=$R6/checkpoint-0040 c120=$R6B/checkpoint-0120"
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo LENGTHMATCH_DONE
