#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$HOME/consensus-attacker}"
WM="${WM:-$HOME/models/qwen3.5-9b}"
BASE="${BASE:-/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft}"
ADAPTER="${ADAPTER:-$REPO/Experiment/Experiment1_Main/runs/exp1_nodefense_qwen3_ppo_dominant_v4/best}"
OUT="${OUT:-$HERE/results/exp5_qwen35_9b_paired_attack_effect_v3}"
N_EXP="${N_EXP:-200}"
DUMP_N="${DUMP_N:-$N_EXP}"
HORIZONS="${HORIZONS:-10,15,20,30,40,50}"
BENIGN_COUNTS="${BENIGN_COUNTS:-4,5,6,7,8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
LOG_EVERY="${LOG_EVERY:-10}"

if [[ $# -gt 0 ]]; then SEEDS=("$@"); else SEEDS=(9000 9001 9002 9003); fi
[[ -s "$WM/config.json" ]] || { echo "[error] missing WM: $WM" >&2; exit 2; }
[[ -s "$BASE/config.json" ]] || { echo "[error] missing SFT base: $BASE" >&2; exit 2; }
[[ -s "$ADAPTER/adapter_model.safetensors" ]] || { echo "[error] missing Exp1 adapter: $ADAPTER" >&2; exit 2; }
mkdir -p "$OUT"

for seed in "${SEEDS[@]}"; do
  json="$OUT/eval_seed${seed}.json"
  log="$OUT/eval_seed${seed}.log"
  if [[ -s "$json" ]]; then echo "[skip] seed=$seed already complete"; continue; fi
  echo "========== EXP5 AGENT SCALING seed=$seed =========="
  python -u "$HERE/eval_exp5_agent_scaling.py" \
    --wm_model_dir "$WM" --llm_model_dir "$BASE" --rl_adapter "$ADAPTER" \
    --benign_counts "$BENIGN_COUNTS" --modes no_attack,ppo \
    --n_exp "$N_EXP" --seed "$seed" --eval_batch_size "$EVAL_BATCH_SIZE" \
    --horizons "$HORIZONS" \
    --wm_gpu_mem_util 0.60 --max_model_len 2048 \
    --wm_max_new_tokens 200 --actor_max_message_tokens 48 --log_every "$LOG_EVERY" \
    --dump_json "$json" --dump_n "$DUMP_N" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "[error] seed=$seed exit_status=$status" >&2
    exit "$status"
  fi
  [[ -s "$json" ]] || { echo "[error] seed=$seed returned without JSON" >&2; exit 3; }
done
