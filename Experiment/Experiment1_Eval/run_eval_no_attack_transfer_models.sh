#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(9000 9001 9002 9003 9004)
fi
SPECS=(
  "llama31_8b:$HOME/models/Meta-Llama-3.1-8B-Instruct"
  "mistral7b_v03:$HOME/models/Mistral-7B-Instruct-v0.3"
  "gemma3_12b:$HOME/models/gemma-3-12b-it"
)
ROOT="results/exp1_sync_no_attack_transfer_models"

for spec in "${SPECS[@]}"; do
  tag="${spec%%:*}"
  model="${spec#*:}"
  if [[ ! -s "$model/config.json" ]]; then
    echo "[error] incomplete or missing model: $model" >&2
    exit 1
  fi
  out="$ROOT/$tag"
  mkdir -p "$out"
  for seed in "${SEEDS[@]}"; do
    json="$out/eval_seed${seed}.json"
    log="$out/eval_seed${seed}.log"
    if [[ -s "$json" ]]; then
      echo "[skip] $tag seed=$seed already complete"
      continue
    fi
    echo "========== NO ATTACK $tag SEED $seed =========="
    python -u eval_no_attacker_consensus.py \
      --wm_model_dir "$model" --wm_backend vllm \
      --n_exp 200 --seed "$seed" --eval_batch_size 16 \
      --wm_gpu_mem_util 0.55 --max_model_len 2048 \
      --wm_max_new_tokens 200 --log_every 0 \
      --dump_json "$json" --dump_n 200 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    if [[ $status -ne 0 ]]; then
      echo "[error] $tag seed=$seed exit_status=$status" >&2
      exit "$status"
    fi
  done
done
