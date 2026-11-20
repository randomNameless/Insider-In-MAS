#!/usr/bin/env bash
# Sequential victims, one model load per victim for all conditions/seeds.
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=("$@")
if [ "${#SEEDS[@]}" -eq 0 ]; then SEEDS=(9000 9001 9002 9003 9004); fi
OUT="${OUT:-results/channel_v1}"
EXTRA=()
if [ -n "${ENV_MODULE:-}" ]; then EXTRA+=(--env-module "$ENV_MODULE"); fi
if [ -n "${RL_ADAPTER:-}" ]; then EXTRA+=(--rl_adapter "$RL_ADAPTER"); fi
if [ -n "${SFT_BASE:-}" ]; then EXTRA+=(--llm_model_dir "$SFT_BASE"); fi
for VICTIM in wm qwen35_9b; do
  python eval_channels.py --victim "$VICTIM" --seeds "${SEEDS[@]}" \
    --out "$OUT" "${EXTRA[@]}" --check
done
for VICTIM in wm qwen35_9b; do
  if [ -d "$OUT/$VICTIM" ] && [ -n "$(ls -A "$OUT/$VICTIM")" ]; then
    echo "Output exists: $OUT/$VICTIM. Choose a new OUT; no overwrites." >&2
    exit 1
  fi
done
for VICTIM in wm qwen35_9b; do
  python -u eval_channels.py --victim "$VICTIM" --seeds "${SEEDS[@]}" \
    --n_exp "${N_EXP:-200}" --eval_batch_size "${BATCH_SIZE:-16}" \
    --out "$OUT" "${EXTRA[@]}"
done
python summarize_channels.py --result-dir "$OUT" --seeds "${SEEDS[@]}"
