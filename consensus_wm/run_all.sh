#!/usr/bin/env bash
# Full chain: 9B collection -> A/B/C SFT -> open/closed-loop eval. Approved by 2026-10-31.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$PWD/.venv/bin:$PATH
D=~/data/cwm; R=~/runs
consensus_wm/collect_all.sh
# SFT records: A/B from fixed-seat games, C from shuffled; val = corresponding test set
python -m consensus_wm.build_sft --mode persona --inp $D/fixed_train.jsonl   --out $D/A_train.jsonl
python -m consensus_wm.build_sft --mode persona --inp $D/fixed_test.jsonl    --out $D/A_val.jsonl
python -m consensus_wm.build_sft --mode id      --inp $D/fixed_train.jsonl   --out $D/B_train.jsonl
python -m consensus_wm.build_sft --mode id      --inp $D/fixed_test.jsonl    --out $D/B_val.jsonl
python -m consensus_wm.build_sft --mode id      --inp $D/shuffle_train.jsonl --out $D/C_train.jsonl
python -m consensus_wm.build_sft --mode id      --inp $D/shuffle_test.jsonl  --out $D/C_val.jsonl
for arm in A B C; do
  python -m consensus_wm.sft_train --model ~/models/qwen3-4b --train $D/${arm}_train.jsonl \
    --val $D/${arm}_val.jsonl --out $R/cwm_sft_$arm --epochs 1 > $R/cwm_sft_$arm.log 2>&1
done
E=$R/cwm_eval; mkdir -p $E
# open-loop: 9B self-reference on both test sets, then each arm on both test sets
for s in fixed shuffle; do
  python -m consensus_wm.eval_wm openloop --model ~/models/qwen3.5-9b --prompt ref --inp $D/${s}_test.jsonl --out $E/ref9b_$s.jsonl
  python -m consensus_wm.eval_wm openloop --model $R/cwm_sft_A/best --prompt persona --inp $D/${s}_test.jsonl --out $E/A_$s.jsonl
  for arm in B C; do
    python -m consensus_wm.eval_wm openloop --model $R/cwm_sft_$arm/best --prompt id --inp $D/${s}_test.jsonl --out $E/${arm}_$s.jsonl
  done
done
# closed-loop vs random-jumper attacker, paired seeds, fixed and shuffled seats
for s in fixed shuffle; do
  python -m consensus_wm.collect_9b --model ~/models/qwen3.5-9b --out $E/cl9b_$s.jsonl --n-games 300 --seat-mode $s --seed 21 --llm-attacker 0
  python -m consensus_wm.eval_wm closedloop --model $R/cwm_sft_A/best --prompt persona --seat-mode $s --out $E/clA_$s.json
  for arm in B C; do
    python -m consensus_wm.eval_wm closedloop --model $R/cwm_sft_$arm/best --prompt id --seat-mode $s --out $E/cl${arm}_$s.json
  done
done
echo ALL_DONE
