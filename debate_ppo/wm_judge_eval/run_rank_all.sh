#!/usr/bin/env bash
# Best-of-8 ranker comparison on the DCM WM's offline oracle test (rank_bon.py): debate-wm-v1 + prompted judges.
set -euo pipefail
cd "$(dirname "$0")/../.."
export VLLM_USE_FLASHINFER_SAMPLER=0
PY=python
OUT=runs_local/wm_judge_eval/rank_bon
[ -f $OUT/wm__roleplay_direct.jsonl ] || $PY -m debate_ppo.wm_judge_eval.rank_bon --model artifacts/models/debate-wm-v1 --tag wm --judges roleplay_gen,roleplay_direct
[ -f $OUT/q4b__observer_v2.jsonl ] || $PY -m debate_ppo.wm_judge_eval.rank_bon --model /path/to/models/qwen3.5-4b --tag q4b --judges observer_v2
[ -f $OUT/q9b__observer_v2.jsonl ] || $PY -m debate_ppo.wm_judge_eval.rank_bon --model /path/to/models/qwen3.5-9b --tag q9b --judges observer_v2
$PY -m debate_ppo.wm_judge_eval.rank_bon --metrics
echo RANK_ALL_DONE
