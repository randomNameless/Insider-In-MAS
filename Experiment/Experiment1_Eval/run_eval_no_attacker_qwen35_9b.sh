#!/usr/bin/env bash
# Paired-seed no-attacker control on the Qwen3.5-9B victim/world model.
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WM="$HOME/models/qwen3.5-9b"
OUT="results/no_attacker_qwen35_9b"
if (( $# > 0 )); then
  SEEDS=("$@")
else
  SEEDS=(9000 9001 9002 9003 9004)
fi

if [[ ! -f "$WM/config.json" ]]; then
  echo "Missing Qwen3.5-9B checkpoint: $WM/config.json" >&2
  exit 2
fi

mkdir -p "$OUT"
for seed in "${SEEDS[@]}"; do
  echo "===== NO-ATTACKER QWEN3.5-9B seed=${seed} ====="
  python -u eval_no_attacker_consensus.py \
    --wm_model_dir "$WM" \
    --wm_backend vllm \
    --n_exp 200 \
    --seed "$seed" \
    --eval_batch_size 16 \
    --wm_max_new_tokens 200 \
    --wm_gpu_mem_util 0.65 \
    --log_every 0 \
    --dump_json "$OUT/eval_seed${seed}.json" \
    --dump_n 200 \
    2>&1 | tee "$OUT/eval_seed${seed}.log"
  status=${PIPESTATUS[0]}
  echo "[no-attacker seed=${seed}] exit_status=${status}"
  if (( status != 0 )); then
    exit "$status"
  fi
done

