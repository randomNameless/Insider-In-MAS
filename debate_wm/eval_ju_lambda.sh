#!/usr/bin/env bash
# Literature baseline (2026-10-22): Ju et al. 2025 adaptive psychological persuasion (arXiv 2506.06800),
# debate_ppo/baseline_ju.py + debate_ppo/ju_dpo/. Run after debate_ppo/ju_dpo/run_lambda.sh all.
# Same split (test, n=400) / seeds (3,4,5) / targets (targets_none9b.json) as the other baseline batches; honest = Qwen3.5-9B.
#   ju_q3_4b             persuader Qwen3-4B (2026-10-22), no-strategy prompt          (their "No Strategy, Before")
#   ju_q3_4b__<best>     same, best fixed strategy on the FIT split (best_strategy.txt)      (their static strategy arm)
#   ju_dpo_q3_4b         DPO-trained persuader, no-strategy prompt = adaptive selection     (their method, Table 4 "After")
# Only the Ju arms + none (2026-10-23: same test questions as the earlier batches, other baselines not re-run; every
# delta is paired against this batch's none). ANCHORS=1 also re-runs targeted_4b, kraidia_9b, bozdag_q25_7b in-batch.
# Smoke: JU_DIR=~/runs/ju_dpo_q3_4b_smoke JU_DPO=~/models/ju-dpo-q3-4b-smoke RUN=~/runs/smoke_ju N=5 SEEDS=3 $0
# EXTRA_STRATS="authority_effect fluency_effect ..." adds more fixed-strategy arms (base persuader).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
D=${JU_DIR:-/path/to/runs/ju_dpo_q3_4b}
DPO=${JU_DPO:-$M/ju-dpo-q3-4b}
BEST=$(cat $D/best_strategy.txt)
CONDS="none,ju_q3_4b,ju_q3_4b__$BEST,ju_dpo_q3_4b"
ATT="\"ju_q3_4b\":\"$M/qwen3-4b\",\"ju_q3_4b__$BEST\":\"$M/qwen3-4b\",\"ju_dpo_q3_4b\":\"$DPO\""
if [ -n "${ANCHORS:-}" ]; then
  CONDS="$CONDS,targeted_4b,kraidia_9b,bozdag_q25_7b"
  ATT="$ATT,\"targeted_4b\":\"$M/qwen3-4b\",\"kraidia_9b\":\"$M/qwen3.5-9b\",\"bozdag_q25_7b\":\"$M/qwen2.5-7b-instruct\""
fi
for s in ${EXTRA_STRATS:-}; do
  [ "$s" = "$BEST" ] && continue
  CONDS="$CONDS,ju_q3_4b__$s"; ATT="$ATT,\"ju_q3_4b__$s\":\"$M/qwen3-4b\""
done
RUN=${RUN:-/path/to/runs/exp9_ju_test_9b}
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "ju2025_psych_persuasion_dpo_test_9b_h100" --victim $M/qwen3.5-9b \
    --attackers "{$ATT}" --conds $CONDS \
    --seeds ${SEEDS:-3,4,5} --n ${N:-400} --answer-first --profile-set v1 --max-model-len 16384 \
    --targets /path/to/runs/targets_none9b.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo JU_DONE
