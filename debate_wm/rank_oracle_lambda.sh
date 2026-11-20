#!/usr/bin/env bash
# Oracle-vs-belief-WM ranking experiment (2026-10-29): debate_wm/rank_oracle.py, 4 persuaders x 300 states x 8 candidates,
# 9B oracle K=4, PPOBeliefWM scores, tables in $OUT/RESULTS.md. Every stage is a separate process (one model on the GPU at a time).
# Lambda:  setsid nohup bash ~/consensus-attacker/debate_wm/rank_oracle_lambda.sh > ~/runs/rank_oracle_exp10.log 2>&1 &
# SMOKE=1: 3 questions (9 states), separate dir.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
OUT=${OUT:-/path/to/runs/rank_oracle_exp10}${SMOKE:+_smoke}
RO="python debate_wm/rank_oracle.py"
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
ts "start -> $OUT"
N=${SMOKE:+3}; N=${N:-100}
[ -f $OUT/states.jsonl ] || $RO states --out $OUT --n $N
for spec in dpo=$M/strat-dpo-q3-4b ppo_c120=$M/exp9-v6-tgt-c120 dpo_ppo_sd120=$M/exp9-v6sd-tgt-c120 base_q3_4b=$M/qwen3-4b; do
  ts "gen $spec"; $RO gen --out $OUT --persuader $spec
done
ts "oracle"; $RO oracle --out $OUT
ts "wm";     [ -f $OUT/wm.jsonl ] || $RO wm --out $OUT
ts "analyze"; CUDA_VISIBLE_DEVICES= $RO analyze --out $OUT
ts "RANK_ORACLE_DONE"
