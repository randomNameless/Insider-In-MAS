#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
SFT="${SFT:-/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft}"
LORA="${LORA:-$REPO/Experiment/Experiment1_Main/runs/exp1_nodefense_qwen3_ppo_dominant_v4/best}"
WM="${WM:-$HOME/models/qwen3.5-9b}"
OUT="${OUT:-$HERE/results/exp6_complete_exp5_b5_repro_v6}"
N_EXP="${N_EXP:-200}"
DUMP_N="${DUMP_N:-200}"
BATCH="${BATCH:-8}"
MODES="${MODES:-no_attack,ppo}"
HORIZONS="${HORIZONS:-10,15,20,30,40,50}"
TOPOLOGIES="${TOPOLOGIES:-complete}"

mkdir -p "$OUT"
if [[ "$#" -eq 0 ]]; then
  set -- 9000
fi

for seed in "$@"; do
  json="$OUT/eval_seed${seed}.json"
  log="$OUT/eval_seed${seed}.log"
  if [[ -s "$json" ]]; then
    echo "[skip] $json exists"
    continue
  fi
  python -u "$HERE/eval_exp6_topology.py" \
    --wm_model_dir "$WM" \
    --llm_model_dir "$SFT" \
    --rl_adapter "$LORA" \
    --topologies "$TOPOLOGIES" \
    --modes "$MODES" --horizons "$HORIZONS" --log_every 40 \
    --seed "$seed" --topology_seed "$seed" \
    --n_exp "$N_EXP" --dump_n "$DUMP_N" \
    --eval_batch_size "$BATCH" \
    --dump_json "$json" 2>&1 | tee "$log"
done

python "$HERE/summarize_exp6.py" --result_dir "$OUT" --seeds "$@"
