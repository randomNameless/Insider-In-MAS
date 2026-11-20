#!/usr/bin/env bash
# Strategy-catalog DPO attacker (2026-10-23), debate_ppo/strat_dpo/. Run after debate_ppo/strat_dpo/run.sh all.
# Same split (test, n=400) / seeds (3,4,5) / targets (targets_none9b.json) as the v6, Ju and literature-baseline batches;
# honest = Qwen3.5-9B. Training used fit (pairs) and val (held-out pairs) only.
#   strat_q3_4b           base Qwen3-4B, no-strategy prompt (the "before" arm)
#   strat_q3_4b__<best>   base Qwen3-4B + the catalog strategy with the highest mean WM score on FIT (best_strategy.txt)
#   stratdpo_q3_4b        DPO model, no-strategy prompt
# The opponent model is written every round by base Qwen3-4B (as in the DPO data). Every delta is paired against this
# batch's none. ANCHORS=1 also re-runs ju_dpo_q3_4b, targeted_model_rl_c040 (v6) and kraidia_9b in-batch.
# STRAT_MAX_TOKENS=500 (2026-10-24): message cap for the eval only (the DPO data used 400); NO_FIXED=1 drops the D1 arm.
# Smoke: STRAT_DIR=~/runs/strat_dpo_q3_4b_smoke STRAT_DPO=~/models/strat-dpo-q3-4b-smoke RUN=~/runs/smoke_strat_eval N=5 SEEDS=3 $0
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
D=${STRAT_DIR:-/path/to/runs/strat_dpo_q3_4b}
DPO=${STRAT_DPO:-$M/strat-dpo-q3-4b}
BEST=$(cat $D/best_strategy.txt)
CONDS="none,strat_q3_4b,strat_q3_4b__$BEST,stratdpo_q3_4b"
ATT="\"strat_q3_4b\":\"$M/qwen3-4b\",\"strat_q3_4b__$BEST\":\"$M/qwen3-4b\",\"stratdpo_q3_4b\":\"$DPO\""
if [ -n "${NO_FIXED:-}" ]; then        # drop the fixed-strategy arm
  CONDS="none,strat_q3_4b,stratdpo_q3_4b"; ATT="\"strat_q3_4b\":\"$M/qwen3-4b\",\"stratdpo_q3_4b\":\"$DPO\""
fi
if [ -n "${ANCHORS:-}" ]; then
  CONDS="$CONDS,ju_dpo_q3_4b,targeted_model_rl_c040,kraidia_9b"
  ATT="$ATT,\"ju_dpo_q3_4b\":\"$M/ju-dpo-q3-4b\",\"targeted_model_rl_c040\":\"$M/exp9-v6-tgt-c040\",\"kraidia_9b\":\"$M/qwen3.5-9b\""
fi
RUN=${RUN:-/path/to/runs/exp9_strat_test_9b}
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "strat_catalog_dpo_test_9b_h100" --victim $M/qwen3.5-9b \
    --attackers "{$ATT}" --conds $CONDS --strat-om-model $M/qwen3-4b --strat-max-tokens ${STRAT_MAX_TOKENS:-400} \
    --seeds ${SEEDS:-3,4,5} --n ${N:-400} --answer-first --profile-set v1 --max-model-len 16384 --arg-tokens 200 \
    --targets /path/to/runs/targets_none9b.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo STRAT_DONE
