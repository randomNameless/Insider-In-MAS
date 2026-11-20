#!/usr/bin/env bash
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT="$HERE/results/exp2_paper_wmsr_v1"
BASE="/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft"
ADAPTER="$HOME/consensus-attacker/Experiment/Experiment2_WMSR/runs/exp2_paper_wmsr_qwen3_ppo_v4_lr2e5_v1/best"
DQN="$HERE/runs/dqn_paper_wmsr_v1/attacker_softq_wmsr.pt"
WM="$HOME/models/qwen35-wm"
if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=(9000 9001 9002 9003 9004)
fi

mkdir -p "$OUT"
for seed in "${SEEDS[@]}"; do
  python -u eval_exp2_wmsr.py \
    --wmsr_F 1 \
    --wm_model_dir "$WM" --wm_backend vllm \
    --llm_model_dir "$BASE" --rl_adapter "$ADAPTER" \
    --dqn_ckpt "$DQN" \
    --attacks random,extreme,prompt,dqn,model_base,model_rl \
    --n_exp 200 --seed "$seed" --eval_batch_size 16 \
    --max_new_tokens 48 --wm_max_new_tokens 200 \
    --wm_gpu_mem_util 0.65 --log_every 0 \
    --dump_json "$OUT/eval_seed${seed}.json" --dump_n 200 \
    2>&1 | tee "$OUT/eval_seed${seed}.log"
  status=${PIPESTATUS[0]}
  (( status == 0 )) || exit "$status"
done
