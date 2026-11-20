#!/usr/bin/env bash
# WM-vs-prompt-judge evaluation on the local RTX 5090 , conda env `insider`.
# 1. eval set from test-split 9B debate logs; 2. every judge x model; 3. trivial baselines; 4. metrics.
# Re-runnable: finished prediction files are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
export VLLM_USE_FLASHINFER_SAMPLER=0
PY=python
WM=artifacts/models/debate-wm-v1      # debate world model checkpoint
Q4=/path/to/models/qwen3.5-4b                        # the WM's base (untrained)
Q9=/path/to/models/qwen3.5-9b                        # the honest agent itself
OUT=runs_local/wm_judge_eval
REC=$OUT/eval_turns.jsonl
[ -f $REC ] || $PY -m debate_ppo.wm_judge_eval.build_eval_set --out $REC

run() {  # tag model judges...
  local tag=$1 model=$2; shift 2
  local todo=()
  for j in "$@"; do [ -f $OUT/preds/${tag}__$j.jsonl ] || todo+=($j); done
  [ ${#todo[@]} -eq 0 ] && { echo "skip $tag"; return; }
  $PY -m debate_ppo.wm_judge_eval.run_judge --model $model --judges $(IFS=,; echo "${todo[*]}") \
    --records $REC --out-prefix $OUT/preds/$tag
}
run wm   $WM roleplay_gen roleplay_direct observer_v1
run q4b  $Q4 roleplay_gen roleplay_direct observer_v1 observer_v2 observer_cot observer_yesno
run q9b  $Q9 roleplay_gen roleplay_direct observer_v1 observer_v2 observer_cot observer_yesno
$PY -m debate_ppo.wm_judge_eval.trivial --records $REC --out-prefix $OUT/preds/trivial
$PY -m debate_ppo.wm_judge_eval.metrics --records $REC --preds $OUT/preds --out $OUT/results
echo WM_JUDGE_EVAL_DONE
