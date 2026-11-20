#!/usr/bin/env bash
set -euo pipefail
for SPEC in "wm:$HOME/models/qwen35-wm" "qwen35_9b:$HOME/models/qwen3.5-9b"; do
  TAG="${SPEC%%:*}"; MODEL="${SPEC#*:}"; OUT="results/exp4_sync_no_attacker_sac_v2_vllm/$TAG"
  mkdir -p "$OUT"
  for SEED in 9000 9001 9002 9003 9004; do
    JSON="$OUT/eval_seed${SEED}.json"
    if [[ -s "$JSON" ]] && python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["summary"]["episodes"] == 200' "$JSON" 2>/dev/null; then
      echo "[skip] completed $JSON"
      continue
    fi
    python -u eval_exp4_no_attacker.py --model_dir "$MODEL" --n_exp 200 \
      --backend vllm --gpu_mem_util 0.75 --batch_size 16 --seed "$SEED" \
      --log_every 50 --dump_json "$JSON" \
      2>&1 | tee "$OUT/eval_seed${SEED}.log"
  done
done
