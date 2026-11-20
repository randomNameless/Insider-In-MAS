#!/usr/bin/env bash
# DCM belief WM with a Qwen3-4B stance judge instead of Qwen2.5-1.5B (2026-10-29): same recipe as ppo_wm_driver.sh /
# ppo_pre_driver.sh (abcd --zero-emb, d_ranker --train-only --save) on the re-scored messages, then the same judge eval
# (run_dcm.py, 4 variants) and the same Best-of-8 ranker test (rank_dcm.py) as the 1.5B DCM. Idempotent: finished files skipped.
# Step 0 (not here, launched first, ~6 h): DCM_JUDGE=Qwen/Qwen3-4B DCM_JUDGE_REV=$REV DCM_STANCE_SUBDIR=q3_4b python -m mmlupro.ppo_stance
#   -> dcm_analysis/data/mmlupro_ppo/stance/q3_4b/scores_all.jsonl (log results/abcd/ppo_stance_q3_4b.log)
set -euo pipefail
REPO=/path/to/repo
DCM=/path/to/latent_agent_transfer/dcm_analysis
PY=python
JUDGE=Qwen/Qwen3-4B
REV=1cfa9a7208912126459214e8b04321603b3df60c
SUF=_q3_4b
EV=$REPO/runs_local/wm_judge_eval
cd $DCM
export DCM_PREFIX=mmluproppo_ DCM_FIT_DATASETS=mmlupro PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DCM_SPLIT_FILE=$DCM/data/mmlupro_ppo/split.json
log() { echo "[dcm4b $(date +%T)] $*"; }

# 0. wait for the stance scores
SC=$DCM/data/mmlupro_ppo/stance/q3_4b/scores_all.jsonl
until [ -f $SC ] && grep -q "^scored 254496 messages" $DCM/results/abcd/ppo_stance_q3_4b.log; do sleep 300; done
N=$(wc -l < $SC); log "stance scores: $N lines"; [ "$N" -eq 254496 ] || { log STANCE_INCOMPLETE; exit 1; }
# d_ranker reads stance/scores<SUF>_std.jsonl (STANCE_SUFFIX convention of the fine-tuned-judge runs; these are raw log-odds,
# not standardised, exactly like scores_all.jsonl of the 1.5B judge); the pre-readout data dir shares the same messages.
ln -sf $SC $DCM/data/mmlupro_ppo/stance/scores${SUF}_std.jsonl
ln -sf $SC $DCM/data/mmlupro_ppo_pre/stance/scores${SUF}_std.jsonl

# 1. fit: abcd (A/B/D test KL, same report as the 1.5B fit) + d_ranker --save (the D ensemble the selector loads)
for v in post pre; do
  DD=data/mmlupro_ppo; T=_ppo; [ $v = pre ] && { DD=data/mmlupro_ppo_pre; T=_ppo_pre; }
  if [ ! -f $DCM/results/abcd_$(basename $DD)/abcd_${T}${SUF}.json ]; then
    log "abcd $v"; DCM_DATA=$DD $PY -u -m mmlupro.abcd --zero-emb --tag ${T}${SUF} --stance-file scores${SUF}_std.jsonl > results/abcd/ppo${T#_ppo}_abcd${SUF}.log 2>&1 || { log ABCD_FAILED; exit 1; }
  fi
  if [ ! -f $DCM/results/abcd_$(basename $DD)/d_model${SUF}.pt ]; then
    log "d_ranker $v"; DCM_DATA=$DD STANCE_SUFFIX=$SUF $PY -u -m mmlupro.d_ranker --train-only --save > results/abcd/ppo${T#_ppo}_d_ranker_train${SUF}.log 2>&1 || { log DRANKER_FAILED; exit 1; }
  fi
done

# 2. judge eval on the 7,920 held-out turns (rounds 2-4), 4 variants
for v in post pre; do for m in wb tb; do
  f=$EV/preds/dcm__dcm4b_${v}_${m}.jsonl
  [ -f $f ] && { log "skip $f"; continue; }
  log "run_dcm $v $m"
  $PY $REPO/debate_ppo/wm_judge_eval/run_dcm.py --mode $m --variant $v --judge $JUDGE --judge-rev $REV --model-suffix $SUF --judge-tag 4b --out-prefix $EV/preds/dcm > $EV/run_dcm4b_${v}_${m}.log 2>&1 || { log RUN_DCM_FAILED; exit 1; }
done; done

# 3. Best-of-8 ranker test (DCM's own sd120 oracle set)
for v in post pre; do
  f=$EV/rank_bon/dcm4b__WM_${v}.jsonl
  [ -f $f ] && { log "skip $f"; continue; }
  log "rank_dcm $v"
  $PY $REPO/debate_ppo/wm_judge_eval/rank_dcm.py --variant $v --tag dcm4b --judge $JUDGE --judge-rev $REV --model-suffix $SUF > $EV/rank_bon/dcm4b_${v}.log 2>&1 || { log RANK_DCM_FAILED; exit 1; }
done

# 4. metrics (insider env, CPU): main table, paired diffs, ranker table
cd $REPO; PYI=python
$PYI -m debate_ppo.wm_judge_eval.metrics --records $EV/eval_turns.jsonl --preds $EV/preds --out $EV/results > $EV/metrics_dcm4b.log 2>&1 || { log METRICS_FAILED; exit 1; }
$PYI -m debate_ppo.wm_judge_eval.paired --records $EV/eval_turns.jsonl --preds $EV/preds --t-min 1 --n-boot 1000 --out $EV/paired_dcm4b.json \
  --pairs dcm__dcm4b_post_wb:dcm__dcm_post_wb,dcm__dcm4b_post_tb:dcm__dcm_post_tb,dcm__dcm4b_pre_wb:dcm__dcm_pre_wb,dcm__dcm4b_pre_tb:dcm__dcm_pre_tb,q4b__observer_v2:dcm__dcm4b_post_wb,wm__roleplay_direct:dcm__dcm4b_post_wb,q9b__observer_v2:dcm__dcm4b_post_wb,q4b__observer_v2:dcm__dcm4b_post_tb \
  > $EV/paired_dcm4b.log 2>&1 || { log PAIRED_FAILED; exit 1; }
$PYI -m debate_ppo.wm_judge_eval.rank_bon --metrics > $EV/rank_bon/rank_metrics_dcm4b.log 2>&1 || { log RANK_METRICS_FAILED; exit 1; }
log DCM4B_ALL_DONE
