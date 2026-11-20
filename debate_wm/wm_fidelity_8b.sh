#!/usr/bin/env bash
# How well do EXISTING models predict the real Qwen3-8B honest agents? (no training)
# Reference = the 8B's own turns in the finished 8B evaluation (test 400 x 3 seeds).
# Candidates: debate-wm-v1 (distilled to imitate the 9B), untrained Qwen3.5-4B (its student base),
# untrained Qwen3-4B, the old opd-mm WM, and the 8B itself resampled (ceiling).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
R=/path/to/data/wm/v1_test8b.jsonl
[ -f $R ] || python -m debate_wm.build_sft --states /path/to/runs/exp9_eval_honest_qwen3_8b/state.json --out $R
mkdir -p /path/to/runs/wm_eval_8b
for X in wm_v1_9bdistill:$M/debate-wm-v1 base_qwen35_4b:$M/qwen3.5-4b base_qwen3_4b:$M/qwen3-4b old_opdmm:$M/qwen35-wm ceiling_8b:$M/qwen3-8b; do
  N=${X%%:*}; P=${X#*:}
  [ -f /path/to/runs/wm_eval_8b/$N.json ] && continue
  python -m debate_wm.eval_wm --model $P --records $R --n 12000 --seed 0 \
    --out /path/to/runs/wm_eval_8b/$N.json > /path/to/runs/wm_eval_8b/$N.log 2>&1
done
echo FID8B_DONE
