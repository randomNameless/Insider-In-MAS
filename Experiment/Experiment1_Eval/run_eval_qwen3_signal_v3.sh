#!/usr/bin/env bash
# Multi-seed evaluation isolated to results/qwen3_signal_v3.
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

OUT_DIR="results/qwen3_signal_v3"
ACTOR_BASE="/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft"
RL_ADAPTER="$HOME/consensus-attacker/Experiment/Experiment1_Main/runs/exp1_nodefense_qwen3_signal_v3/best"
DQN_CKPT="$HERE/attacker_softq_qwen_mixed.pt"
if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=(9000 9001 9002 9003 9004)
fi

mkdir -p "$OUT_DIR"
for seed in "${SEEDS[@]}"; do
  echo "===== QWEN3 SIGNAL V3 EVAL seed=${seed} ====="
  python -u eval_attacks_consensus.py \
    --wm_model_dir "$HOME/models/qwen35-wm" \
    --wm_backend vllm \
    --llm_model_dir "$ACTOR_BASE" \
    --rl_adapter "$RL_ADAPTER" \
    --dqn_ckpt "$DQN_CKPT" \
    --attacks random,extreme,prompt,dqn,model_base,model_rl \
    --n_exp 200 \
    --seed "$seed" \
    --eval_batch_size 16 \
    --max_new_tokens 48 \
    --wm_max_new_tokens 200 \
    --wm_gpu_mem_util 0.65 \
    --log_every 0 \
    --dump_json "$OUT_DIR/eval_seed${seed}.json" \
    --dump_n 200 \
    2>&1 | tee "$OUT_DIR/eval_seed${seed}.log"
  status=${PIPESTATUS[0]}
  echo "[eval seed=${seed}] exit_status=${status}"
done

