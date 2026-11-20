#!/usr/bin/env bash
# Strategy-catalog DPO attacker (2026-10-23): states -> opponent model -> 27 strategy messages -> debate-WM
# scoring -> pairs -> LoRA DPO (ju_dpo/train_dpo.py, same recipe as the Ju et al. arm) -> merge.
# Defaults = Lambda paths. Every stage records the sha256 of the weights it used (config.json), so runs on machines
# with different paths can be checked against each other.
# Usage: run.sh data|train|merge|all      SMOKE=1: 20 questions per split, 1 seed, 2 DPO steps, separate output dir
set -euo pipefail
CA=${CA:-$HOME/consensus-attacker}
M=${M:-$HOME/models}
PM=${PERSUADER:-$M/qwen3-4b}
WM=${WM:-$M/debate-wm-v1}
COLLECT=${COLLECT:-$HOME/runs/wm_collect_v1_fit/state.json,$HOME/runs/wm_collect_v1_val/state.json}  # fit = train, val = held-out
TARGETS=${TARGETS:-$HOME/runs/targets_9b_fitval.json}
OUT=${OUT:-$HOME/runs/strat_dpo_q3_4b}${SMOKE:+_smoke}
MERGED=${MERGED:-$M/strat-dpo-q3-4b}${SMOKE:+-smoke}
VLLM_ENV=${VLLM_ENV:-$CA/.venv/bin/activate}       # vLLM / transformers 5.x
DPO_ENV=${DPO_ENV:-$HOME/venvs/ju-dpo/bin/activate} # trl 0.17 / transformers 4.51 (ju_dpo/install_lambda.sh)
mkdir -p $OUT
cd $CA
MD="python -m debate_ppo.strat_dpo.make_data"

data() {
  [ -f "$VLLM_ENV" ] && source $VLLM_ENV
  [ -f $OUT/states.jsonl ]   || $MD states --out $OUT --collect $COLLECT --targets $TARGETS ${SMOKE:+--n 20} 2>&1 | tee $OUT/states.log
  [ -f $OUT/opponent.jsonl ] || $MD opponent --out $OUT --model $PM 2>&1 | tee $OUT/opponent.log
  [ -f $OUT/persuade.jsonl ] || $MD persuade --out $OUT --model $PM 2>&1 | tee $OUT/persuade.log
  [ -f $OUT/score.done ]     || { $MD score --out $OUT --model $WM 2>&1 | tee -a $OUT/score.log && touch $OUT/score.done; }
  $MD pairs --out $OUT --model $PM 2>&1 | tee $OUT/pairs.log
  [ -f "$VLLM_ENV" ] && deactivate || true
}
train() {
  [ -f "$DPO_ENV" ] && source $DPO_ENV
  python $CA/debate_ppo/ju_dpo/train_dpo.py --model $PM --train-data $OUT/preference_data.jsonl \
    --eval-data $OUT/preference_eval.jsonl --out $OUT/checkpoint ${SMOKE:+--max-steps 2 --eval-steps 1} 2>&1 | tee $OUT/train.log
  [ -f "$DPO_ENV" ] && deactivate || true
}
merge() {
  [ -f "$DPO_ENV" ] && source $DPO_ENV
  python $CA/debate_ppo/ju_dpo/merge.py --base $PM --adapter $OUT/checkpoint --out $MERGED 2>&1 | tee $OUT/merge.log
  ( cd $MERGED && sha256sum *.safetensors ) > $OUT/merged_sha256.txt
  [ -f "$DPO_ENV" ] && deactivate || true
}
case "${1:-}" in
  data) data ;; train) train ;; merge) merge ;;
  all) data; train; merge ;;
  *) echo "usage: $0 data|train|merge|all"; exit 1 ;;
esac
