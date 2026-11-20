#!/usr/bin/env bash
# Regularised re-training of the v6-format strat-DPO (2026-10-30, "token上限增加到500, 然后1"):
# same preference pairs as ~/runs/strat_dpo_v6_q3_4b (4,570 train / 502 held-out, NOT regenerated), base plain Qwen3-4B,
# LoRA r128/a128, 1 epoch, eff. batch 16 as before; changed: LR 1e-4 -> 2e-5, beta 0.1 -> 0.3, + RPO alpha 1.0 (NLL on chosen).
# Usage: run_v6r.sh train|merge|all     SMOKE=1: 2 steps, separate dirs
set -euo pipefail
CA=${CA:-$HOME/consensus-attacker}; M=${M:-$HOME/models}; R=${R:-$HOME/runs}
PM=${PERSUADER:-$M/qwen3-4b}
SRC=$R/strat_dpo_v6_q3_4b
OUT=${OUT:-$R/strat_dpo_v6r_q3_4b}${SMOKE:+_smoke}
MERGED=${MERGED:-$M/strat-dpo-v6r-q3-4b}${SMOKE:+-smoke}
LR=${LR:-2e-5}; BETA=${BETA:-0.3}; RPO=${RPO:-1.0}
DPO_ENV=$HOME/venvs/ju-dpo/bin/activate
mkdir -p $OUT; cd $CA
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
train() {
  source $DPO_ENV
  for f in preference_data.jsonl preference_eval.jsonl pairs_summary.json; do [ -e $OUT/$f ] || cp $SRC/$f $OUT/$f; done
  ( cd $SRC && sha256sum preference_data.jsonl preference_eval.jsonl ) > $OUT/pairs_sha256_from_v6.txt
  ts "dpo-reg lr=$LR beta=$BETA rpo_alpha=$RPO"
  python $CA/debate_ppo/ju_dpo/train_dpo_reg.py --model $PM --train-data $OUT/preference_data.jsonl \
    --eval-data $OUT/preference_eval.jsonl --out $OUT/checkpoint --lr $LR --beta $BETA --rpo-alpha $RPO \
    ${SMOKE:+--max-steps 2 --eval-steps 1} 2>&1 | tee $OUT/train.log
  deactivate || true
}
merge() {
  source $DPO_ENV
  ts "merge"
  python $CA/debate_ppo/ju_dpo/merge.py --base $PM --adapter $OUT/checkpoint --out $MERGED 2>&1 | tee $OUT/merge.log
  ( cd $MERGED && sha256sum *.safetensors ) > $OUT/merged_sha256.txt
  deactivate || true
}
case "${1:-}" in
  train) train ;; merge) merge ;; all) train; merge; ts "RUN_V6R_DONE" ;;
  *) echo "usage: $0 train|merge|all"; exit 1 ;;
esac
