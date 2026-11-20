#!/usr/bin/env bash
# After SFT: export best -> vLLM layout, then turn-level accuracy vs the 9B teacher on test turns for
# the new WM, the old WM (opd-mm) and the untrained Qwen3.5-4B (same 12k stratified turns, same seed).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
until grep -q "\] done" /path/to/runs/wm_sft_v1.log; do sleep 60; done
python -m debate_wm.export_vllm --trained /path/to/runs/wm_sft_v1/best --base /path/to/models/qwen3.5-4b --out /path/to/models/debate-wm-v1
mkdir -p /path/to/runs/wm_eval
for X in new:/path/to/models/debate-wm-v1 old_opdmm:/path/to/models/qwen35-wm base_qwen35_4b:/path/to/models/qwen3.5-4b; do
  NAME=${X%%:*}; M=${X#*:}
  [ -f /path/to/runs/wm_eval/$NAME.json ] && continue
  python -m debate_wm.eval_wm --model $M --records /path/to/data/wm/v1_test9b.jsonl --n 12000 --seed 0 \
    --out /path/to/runs/wm_eval/$NAME.json > /path/to/runs/wm_eval/$NAME.log 2>&1
done
echo WMEVAL_DONE
