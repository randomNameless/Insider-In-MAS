#!/usr/bin/env bash
# Strategy-catalog DPO in the v6 (targeted PPO) format (2026-10-28; plan ~/runs/ppo_diag_20261028/STRATDPO_V6_PLAN.md).
# collect (Qwen3.5-9B `none`, profile v1nn, fit+val, seed 0) -> targets (make_targets rule, per split, merged)
# -> states (rounds 0,1,2) -> gen (6 strategy-hinted + 2 plain v6 candidates per state) -> score (debate-wm-v1, K=2)
# -> pairs (plain v6 prompt, raw argument completion, train subsampled to 4,796) -> LoRA DPO (ju_dpo/train_dpo.py,
# original strat-DPO hyper-parameters, base plain Qwen3-4B) -> merge. Existing strat_dpo files are not modified.
# Usage: run_v6.sh collect|data|train|merge|all      SMOKE=1: 20 questions per split, 2 DPO steps, separate dirs
set -euo pipefail
CA=${CA:-$HOME/consensus-attacker}
M=${M:-$HOME/models}
R=${R:-$HOME/runs}
PM=${PERSUADER:-$M/qwen3-4b}
WM=${WM:-$M/debate-wm-v1}
PS=v1nn
OUT=${OUT:-$R/strat_dpo_v6_q3_4b}${SMOKE:+_smoke}
COL=$R/stratdpo6_collect_${PS}${SMOKE:+_smoke}
MERGED=${MERGED:-$M/strat-dpo-v6-q3-4b}${SMOKE:+-smoke}
VLLM_ENV=$CA/.venv/bin/activate
DPO_ENV=$HOME/venvs/ju-dpo/bin/activate
mkdir -p $OUT
cd $CA
MD="python -m debate_ppo.strat_dpo.make_data_v6"
ts() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

collect() {
  source $VLLM_ENV
  for SPLIT in fit val; do
    if [ -n "${SMOKE:-}" ]; then N=20; elif [ $SPLIT = fit ]; then N=1441; else N=160; fi
    RUN=${COL}_$SPLIT
    ts "collect $SPLIT n=$N -> $RUN"
    [ -f $RUN/state.json ] || python -m debate_ppo.eval_debate init --run $RUN --label "stratdpo6_teacher_9b_${PS}_$SPLIT" \
      --split $SPLIT --n $N --victim $M/qwen3.5-9b --attackers '{}' --conds none --seeds 0 --answer-first --profile-set $PS
    python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
    python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
    python -m debate_wm.make_targets $RUN/state.json --out $RUN/targets.json
  done
  python - ${COL}_fit/targets.json ${COL}_val/targets.json $OUT/targets_${PS}_fitval.json <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
assert not set(a["targets"]) & set(b["targets"])
json.dump({"source": [a["source"], b["source"]], "rule": a["rule"], "n": a["n"] + b["n"],
           "targets": {**a["targets"], **b["targets"]}}, open(sys.argv[3], "w"), indent=1)
print("merged", a["n"] + b["n"], "targets ->", sys.argv[3])
EOF
  deactivate || true
}
data() {
  source $VLLM_ENV
  ts "states";  [ -f $OUT/states.jsonl ] || $MD states --out $OUT --profile-set $PS --targets $OUT/targets_${PS}_fitval.json \
                  --collect ${COL}_fit/state.json,${COL}_val/state.json 2>&1 | tee $OUT/states.log
  ts "gen";     [ -f $OUT/candidates.jsonl ] || $MD gen --out $OUT --profile-set $PS --model $PM 2>&1 | tee $OUT/gen.log
  ts "score";   [ -f $OUT/score.done ] || { $MD score --out $OUT --profile-set $PS --model $WM 2>&1 | tee -a $OUT/score.log && touch $OUT/score.done; }
  ts "pairs";   $MD pairs --out $OUT --profile-set $PS --model $PM 2>&1 | tee $OUT/pairs.log
  deactivate || true
}
train() {
  source $DPO_ENV
  ts "dpo"
  python $CA/debate_ppo/ju_dpo/train_dpo.py --model $PM --train-data $OUT/preference_data.jsonl \
    --eval-data $OUT/preference_eval.jsonl --out $OUT/checkpoint ${SMOKE:+--max-steps 2 --eval-steps 1} 2>&1 | tee $OUT/train.log
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
  collect) collect ;; data) data ;; train) train ;; merge) merge ;;
  all) collect; data; train; merge; ts "RUN_V6_DONE" ;;
  *) echo "usage: $0 collect|data|train|merge|all"; exit 1 ;;
esac
