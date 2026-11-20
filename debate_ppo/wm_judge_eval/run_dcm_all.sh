#!/usr/bin/env bash
# DCM belief WM (latent_agent_transfer) on the judge eval set: WM_post / WM_pre x wb (tracked beliefs) / tb (true beliefs).
# Runs in the dcm_analysis environment (Debate venv), from dcm_analysis (module paths). Finished files are skipped.
set -euo pipefail
REPO=/path/to/repo
PY=python
cd /path/to/latent_agent_transfer/dcm_analysis
for v in post pre; do for m in wb tb; do
  f=$REPO/runs_local/wm_judge_eval/preds/dcm__dcm_${v}_${m}.jsonl
  [ -f $f ] && { echo "skip $f"; continue; }
  echo "[$(date +%T)] dcm $v $m"
  $PY $REPO/debate_ppo/wm_judge_eval/run_dcm.py --mode $m --variant $v --out-prefix $REPO/runs_local/wm_judge_eval/preds/dcm
done; done
echo DCM_ALL_DONE
