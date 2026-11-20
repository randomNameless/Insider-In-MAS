#!/usr/bin/env bash
# 9B data for the persona-masked WM arms: fixed / shuffled seats, train + test (disjoint seeds).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$PWD/.venv/bin:$PATH
M=~/models/qwen3.5-9b; D=~/data/cwm
for spec in "fixed train 2000 1" "shuffle train 2000 2" "fixed test 300 11" "shuffle test 300 12"; do
  set -- $spec
  python -m consensus_wm.collect_9b --model $M --out $D/${1}_${2}.jsonl --n-games $3 --seat-mode $1 --seed $4
done
echo COLLECT_DONE
