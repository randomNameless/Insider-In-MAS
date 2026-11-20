#!/usr/bin/env bash
# Prompt judge WITHOUT the persona (2026-10-29): observer_v2_nopersona with Qwen3.5-9B and Qwen3.5-4B on the 7,920
# held-out turns and on the Best-of-8 ranker test, then the tables again. Waits for the DCM-4B chain (GPU) to finish.
set -euo pipefail
cd "$(dirname "$0")/../.."
export VLLM_USE_FLASHINFER_SAMPLER=0
PY=python
Q4=/path/to/models/qwen3.5-4b
Q9=/path/to/models/qwen3.5-9b
EV=runs_local/wm_judge_eval
REC=$EV/eval_turns.jsonl
J=observer_v2_nopersona
log() { echo "[nopersona $(date +%T)] $*"; }
until grep -qE "DCM4B_ALL_DONE|FAILED|INCOMPLETE" $EV/run_dcm4b_all.log 2>/dev/null; do sleep 120; done
grep -q DCM4B_ALL_DONE $EV/run_dcm4b_all.log || log "WARNING: dcm4b chain did not finish cleanly; running anyway"
for t in q9b q4b; do
  M=$Q9; [ $t = q4b ] && M=$Q4
  [ -f $EV/preds/${t}__$J.jsonl ] || { log "judge $t"; $PY -m debate_ppo.wm_judge_eval.run_judge --model $M --judges $J --records $REC --out-prefix $EV/preds/$t || { log JUDGE_FAILED; exit 1; }; }
  [ -f $EV/rank_bon/${t}__$J.jsonl ] || { log "rank $t"; $PY -m debate_ppo.wm_judge_eval.rank_bon --model $M --tag $t --judges $J || { log RANK_FAILED; exit 1; }; }
done
$PY -m debate_ppo.wm_judge_eval.metrics --records $REC --preds $EV/preds --out $EV/results > $EV/metrics_nopersona.log 2>&1 || { log METRICS_FAILED; exit 1; }
$PY -m debate_ppo.wm_judge_eval.paired --records $REC --preds $EV/preds --t-min 1 --n-boot 1000 --out $EV/paired_nopersona.json \
  --pairs q9b__observer_v2:q9b__$J,q4b__observer_v2:q4b__$J,q9b__$J:dcm__dcm_post_wb,q4b__$J:dcm__dcm_post_wb,q9b__$J:dcm__dcm4b_post_wb,q4b__$J:dcm__dcm4b_post_wb,q9b__$J:wm__roleplay_direct \
  > $EV/paired_nopersona.log 2>&1 || { log PAIRED_FAILED; exit 1; }
$PY -m debate_ppo.wm_judge_eval.rank_bon --metrics > $EV/rank_bon/rank_metrics_nopersona.log 2>&1 || { log RANK_METRICS_FAILED; exit 1; }
log NOPERSONA_ALL_DONE
