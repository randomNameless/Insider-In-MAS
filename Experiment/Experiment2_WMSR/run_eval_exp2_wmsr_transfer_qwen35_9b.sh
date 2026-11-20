#!/usr/bin/env bash
# Sim-to-real transfer evaluation for Experiment 2:
# defense-trained PPO/DQN and all baselines against Qwen3.5-9B benign agents
# protected by the same paper-global W-MSR filter (f=1).
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WM="${WM:-$HOME/models/qwen3.5-9b}"
BASE="${BASE:-/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft}"
ADAPTER="${ADAPTER:-$HERE/runs/exp2_paper_wmsr_qwen3_ppo_v4_lr2e5_v1/best}"
DQN="${DQN:-$HERE/runs/dqn_paper_wmsr_v1/attacker_softq_wmsr.pt}"
OUT="${OUT:-$HERE/results/exp2_paper_wmsr_v1_transfer_qwen35_9b}"

if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=(9000 9001 9002 9003 9004)
fi

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

require_file "$WM/config.json"
require_file "$BASE/config.json"
require_file "$ADAPTER/adapter_config.json"
require_file "$ADAPTER/adapter_model.safetensors"
require_file "$DQN"

mkdir -p "$OUT"
for seed in "${SEEDS[@]}"; do
  echo "===== EXP2 PAPER W-MSR -> QWEN3.5-9B TRANSFER seed=${seed} ====="
  python -u eval_exp2_wmsr.py \
    --wmsr_F 1 \
    --wm_model_dir "$WM" --wm_backend vllm \
    --llm_model_dir "$BASE" --rl_adapter "$ADAPTER" \
    --dqn_ckpt "$DQN" \
    --attacks random,extreme,prompt,dqn,model_base,model_rl \
    --n_exp 200 --seed "$seed" --eval_batch_size 16 \
    --max_new_tokens 48 --wm_max_new_tokens 200 \
    --wm_gpu_mem_util 0.65 --max_model_len 2048 \
    --log_every 0 \
    --dump_json "$OUT/eval_seed${seed}.json" --dump_n 200 \
    2>&1 | tee "$OUT/eval_seed${seed}.log"
  status=${PIPESTATUS[0]}
  echo "[transfer seed=${seed}] exit_status=${status}"
  (( status == 0 )) || exit "$status"
done

echo "===== EXP2 QWEN3.5-9B TRANSFER COMPLETE ====="
grep -hE '^[[:space:]]*attacker=' "$OUT"/eval_seed*.log
