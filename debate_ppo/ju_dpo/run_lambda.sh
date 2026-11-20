#!/usr/bin/env bash
# Ju et al. 2025 adaptive psychological persuasion (DPO) on Lambda, one H100 (2026-10-22).
#   data   strategy sweep on the fit split (persuader Qwen3-4B, listener Qwen3.5-9B; 2026-10-22) -> preference pairs
#   train  LoRA DPO (venv from install_lambda.sh)
#   merge  -> ~/models/ju-dpo-q3-4b
# Usage: run_lambda.sh data|train|merge|all     (SMOKE=1: 5 fit questions, 2 DPO steps, separate output dir)
set -euo pipefail
CA=$HOME/consensus-attacker
M=$HOME/models
PM=${PERSUADER:-$M/qwen3-4b}
OUT=$HOME/runs/ju_dpo_q3_4b${SMOKE:+_smoke}
MERGED=$M/ju-dpo-q3-4b${SMOKE:+-smoke}
mkdir -p $OUT
cd $CA

data() {
  source $CA/.venv/bin/activate
  N=${SMOKE:+--n 5}
  [ -f $OUT/persuade.jsonl ] || python -m debate_ppo.ju_dpo.make_data persuade --out $OUT --model $PM $N 2>&1 | tee $OUT/persuade.log
  [ -f $OUT/listen.jsonl ]   || python -m debate_ppo.ju_dpo.make_data listen --out $OUT --model $M/qwen3.5-9b 2>&1 | tee $OUT/listen.log
  python -m debate_ppo.ju_dpo.make_data pairs --out $OUT 2>&1 | tee $OUT/pairs.log
  deactivate
}
train() {
  source $HOME/venvs/ju-dpo/bin/activate
  python $CA/debate_ppo/ju_dpo/train_dpo.py --model $PM --train-data $OUT/preference_data.jsonl --out $OUT/checkpoint \
    ${SMOKE:+--max-steps 2} 2>&1 | tee $OUT/train.log
  deactivate
}
merge() {
  source $HOME/venvs/ju-dpo/bin/activate
  python $CA/debate_ppo/ju_dpo/merge.py --base $PM --adapter $OUT/checkpoint --out $MERGED 2>&1 | tee $OUT/merge.log
  deactivate
}
case "${1:-}" in
  data) data ;; train) train ;; merge) merge ;;
  all) data; train; merge ;;
  *) echo "usage: $0 data|train|merge|all"; exit 1 ;;
esac
