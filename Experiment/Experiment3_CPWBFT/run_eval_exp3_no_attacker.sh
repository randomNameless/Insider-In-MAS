#!/usr/bin/env bash
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if (( $# > 0 )); then SEEDS=("$@"); else SEEDS=(9000 9001 9002 9003 9004); fi
OUT="${OUT:-$HERE/results/exp3_no_attacker_cpwbft_v1}"
mkdir -p "$OUT"

run_condition() {
  local condition="$1" model="$2" seed="$3" tag="$4"
  [[ -s "$model/config.json" ]] || { echo "Missing: $model/config.json" >&2; return 2; }
  mkdir -p "$OUT/$tag"
  python -u eval_exp3_no_attacker.py \
    --model_dir "$model" --condition "$condition" \
    --n_exp 200 --seed "$seed" --eval_batch_size 8 \
    --temperature 0.7 --max_new_tokens 200 --margin 0 \
    --dump_json "$OUT/$tag/eval_seed${seed}.json" --dump_n 200 \
    2>&1 | tee "$OUT/$tag/eval_seed${seed}.log"
  return "${PIPESTATUS[0]}"
}

for seed in "${SEEDS[@]}"; do
  run_condition "wm_cpwbft_no_attacker" "$HOME/models/qwen35-wm" "$seed" "wm" || exit $?
  run_condition "qwen35_9b_cpwbft_no_attacker" "$HOME/models/qwen3.5-9b" "$seed" "qwen35_9b" || exit $?
done
