#!/usr/bin/env bash
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SEED="${1:-9000}"
N_EXP="${2:-200}"
OUT="$HERE/results/seed${SEED}"
mkdir -p "$OUT"

run_condition() {
  local condition="$1"
  local model="$2"
  local log="$OUT/${condition}.log"
  local json="$OUT/${condition}.json"

  if [[ ! -f "$model/config.json" ]]; then
    echo "Missing model: $model/config.json" >&2
    return 2
  fi

  python -u eval_four_benign.py \
    --model_dir "$model" \
    --condition "$condition" \
    --backend vllm \
    --n_exp "$N_EXP" \
    --seed "$SEED" \
    --max_pos 20 --max_steps 10 --consensus_tol 0 \
    --temperature 0.7 --max_new_tokens 200 \
    --eval_batch_size 16 --gpu_mem_util 0.65 \
    --log_every 0 --dump_json "$json" --dump_n "$N_EXP" \
    2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

# Run sequentially: each invocation tears down its vLLM engine before the next.
run_condition "wm_four_benign" "$HOME/models/qwen35-wm" || exit $?
run_condition "qwen35_9b_four_benign" "$HOME/models/qwen3.5-9b" || exit $?

echo "===== EXPERIMENT 0 SUMMARY ====="
grep -h '^condition=' "$OUT"/*.log

