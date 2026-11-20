#!/usr/bin/env bash
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

TARGET="${1:?usage: $0 wm|qwen35-9b [seed ...]}"
shift
if [[ "$TARGET" == "wm" ]]; then
  WM="${WM:-$HOME/models/qwen35-wm}"
  TAG="wm"
elif [[ "$TARGET" == "qwen35-9b" ]]; then
  WM="${WM:-$HOME/models/qwen3.5-9b}"
  TAG="qwen35_9b_transfer"
else
  echo "target must be wm or qwen35-9b" >&2
  exit 2
fi

BASE="${BASE:-/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft}"
ADAPTER="${ADAPTER:-$HERE/runs/exp3_cpwbft_qwen3_ppo_v4_lr2e5_v1/best}"
OUT="${OUT:-$HERE/results/exp3_cpwbft_v1_${TAG}}"
if (( $# > 0 )); then SEEDS=("$@"); else SEEDS=(9000 9001 9002 9003 9004); fi

for required in "$WM/config.json" "$BASE/config.json" \
  "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors"; do
  [[ -s "$required" ]] || { echo "Missing: $required" >&2; exit 2; }
done

mkdir -p "$OUT"
for seed in "${SEEDS[@]}"; do
  python -u eval_exp3_cpwbft.py \
    --wm_model_dir "$WM" --llm_model_dir "$BASE" --rl_adapter "$ADAPTER" \
    --attacks prompt,model_base,model_rl \
    --n_exp 200 --seed "$seed" --eval_batch_size 8 \
    --actor_max_message_tokens 48 --wm_max_new_tokens 200 --margin 0 \
    --dump_json "$OUT/eval_seed${seed}.json" --dump_n 200 \
    2>&1 | tee "$OUT/eval_seed${seed}.log"
  status=${PIPESTATUS[0]}
  (( status == 0 )) || exit "$status"
done
