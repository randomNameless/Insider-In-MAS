#!/usr/bin/env bash
# Our targeted PPO persuader (v6 / v6b) vs the literature baselines in ONE paired batch (2026-10-22), local RTX 5090,
# honest = real Qwen3.5-9B. Our conditions replicate debate_wm/eval_v6_targeted.sh (Lambda run exp9_v6_targeted_test_9b)
# exactly: none, targeted_4b, targeted_model_base, targeted_model_rl_{c040,c080,c120}; --answer-first, --arg-tokens 200,
# targets_none9b.json, test n=400, seeds 3,4,5. Checkpoints fixed in advance (c040 = v6 step 40, c080/c120 = v6b);
# merged weights identical to Lambda's (sha256 of c040 checked). Added in the same batch: kraidia_4b, kraidia_9b,
# amayuelasbon_4b, bozdag_q25_7b, tomap_3b (prompts / settings as in runs_local/baselines_test_9b and tomap_test_9b).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
Q4=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/* | head -1)
Q9=/path/to/models/qwen3.5-9b
Q7=/path/to/models/qwen2.5-7b-instruct
M=/path/to/models
ATT=$(printf '{"targeted_4b":"%s","targeted_model_base":"%s","targeted_model_rl_c040":"%s","targeted_model_rl_c080":"%s","targeted_model_rl_c120":"%s","kraidia_4b":"%s","kraidia_9b":"%s","amayuelasbon_4b":"%s","bozdag_q25_7b":"%s","tomap_3b":"%s"}' \
  $Q4 $Q4 $M/exp9-v6-tgt-c040 $M/exp9-v6-tgt-c080 $M/exp9-v6-tgt-c120 $Q4 $Q9 $Q4 $Q7 $M/qwen2.5-3b-instruct-tomap)
RUN=runs_local/v6_vs_baselines_test_9b
R=runs_lambda/runs
if [ ! -f $RUN/state.json ]; then
  $PY -m debate_ppo.eval_debate init --run $RUN --label "v6_targeted_ppo_vs_literature_baselines_test_9b_rtx5090" --victim $Q9 \
    --attackers "$ATT" \
    --conds none,targeted_4b,targeted_model_base,targeted_model_rl_c040,targeted_model_rl_c080,targeted_model_rl_c120,kraidia_4b,kraidia_9b,amayuelasbon_4b,bozdag_q25_7b,tomap_3b \
    --seeds 3,4,5 --n 400 --answer-first --profile-set v1 --arg-tokens 200 --max-model-len 16384 \
    --targets $R/targets_none9b.json --tasks lambda2_backup/data/mmlupro_v1/tasks.jsonl \
    --rl-checkpoint "c040=$R/exp9_debate_ppo_v6_targeted/checkpoint-0040 c080=$R/exp9_debate_ppo_v6b_targeted_len200/checkpoint-0080 c120=$R/exp9_debate_ppo_v6b_targeted_len200/checkpoint-0120"
fi
$PY -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
$PY -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo V6_VS_BASELINES_DONE
