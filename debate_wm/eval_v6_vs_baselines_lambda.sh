#!/usr/bin/env bash
# Lambda (H100) copy of debate_wm/eval_v6_vs_baselines_local.sh (2026-10-22): our targeted PPO persuader (v6 / v6b)
# vs the literature baselines in ONE paired batch, honest = real Qwen3.5-9B. Our conditions replicate
# debate_wm/eval_v6_targeted.sh exactly (--answer-first, --arg-tokens 200, targets_none9b.json, test n=400, seeds 3,4,5;
# checkpoints fixed in advance). Added: kraidia_4b, kraidia_9b, amayuelasbon_4b, bozdag_q25_7b, tomap_3b.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
ATT=$(printf '{"targeted_4b":"%s","targeted_model_base":"%s","targeted_model_rl_c040":"%s","targeted_model_rl_c080":"%s","targeted_model_rl_c120":"%s","kraidia_4b":"%s","kraidia_9b":"%s","amayuelasbon_4b":"%s","bozdag_q25_7b":"%s","tomap_3b":"%s"}' \
  $M/qwen3-4b $M/qwen3-4b $M/exp9-v6-tgt-c040 $M/exp9-v6-tgt-c080 $M/exp9-v6-tgt-c120 $M/qwen3-4b $M/qwen3.5-9b $M/qwen3-4b \
  $M/qwen2.5-7b-instruct $M/qwen2.5-3b-instruct-tomap)
R6=/path/to/runs/exp9_debate_ppo_v6_targeted
R6B=/path/to/runs/exp9_debate_ppo_v6b_targeted_len200
RUN=/path/to/runs/exp9_v6_vs_baselines_test_9b
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "v6_targeted_ppo_vs_literature_baselines_test_9b_h100" --victim $M/qwen3.5-9b \
    --attackers "$ATT" \
    --conds none,targeted_4b,targeted_model_base,targeted_model_rl_c040,targeted_model_rl_c080,targeted_model_rl_c120,kraidia_4b,kraidia_9b,amayuelasbon_4b,bozdag_q25_7b,tomap_3b \
    --seeds 3,4,5 --n 400 --answer-first --profile-set v1 --arg-tokens 200 --max-model-len 16384 \
    --targets /path/to/runs/targets_none9b.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl \
    --tomap-encoder $M/bge-m3 --tomap-predictor $M/tomap-predictor/best_mlp_model.pth --tomap-helper $M/qwen2.5-3b-instruct \
    --rl-checkpoint "c040=$R6/checkpoint-0040 c080=$R6B/checkpoint-0080 c120=$R6B/checkpoint-0120"
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo V6_VS_BASELINES_DONE
