#!/usr/bin/env bash
# Restart training from the newest checkpoint-XXXX when the log has not changed for 25 minutes
# (vLLM 0.23 + Qwen3.5 occasionally hangs inside a decode step). Run in its own tmux session.
CFG=$1; RUN=$2; LOG=$3
cd "$(dirname "$0")/.."
while true; do
  sleep 120
  tmux has-session -t train 2>/dev/null || { echo "[watchdog] no train session; exiting"; exit 0; }
  grep -q "\[done\]" "$LOG" && { echo "[watchdog] training finished"; exit 0; }
  age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  if [ $age -gt 1500 ]; then
    CK=$(ls -d $RUN/checkpoint-* | sort | tail -1); STEP=$((10#${CK##*-}))
    echo "[watchdog] $(date -u) log stale ${age}s -> restart from $CK (step $STEP)"
    tmux kill-session -t train; sleep 10
    mv "$LOG" "${LOG%.log}_hang_$(date +%H%M).log"
    python3 - "$RUN/train_metrics.jsonl" "$STEP" <<'PY'
import json, sys
p, k = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(p)]
open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows if r["it"] < k))
PY
    tmux new -d -s train "source .venv/bin/activate && python -u -m debate_ppo.run --config $CFG --resume-adapter $CK --start-it $STEP 2>&1 | tee $LOG"
  fi
done
