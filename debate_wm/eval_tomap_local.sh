#!/usr/bin/env bash
# Literature baseline (2026-10-21): ToMAP (Han, Liu, You; arXiv 2505.22961) released persuader
# HakHan/Qwen2.5-3B-Instruct-ToMAP in its full setting (3 counterclaims + released attitude predictor on bge-m3),
# debate_ppo/baseline_tomap.py. Honest = real Qwen3.5-9B on the local RTX 5090.
# Same split (test, n=400) / seeds (3,4,5) / targets (targets_none9b.json) as runs_local/baselines_test_9b;
# none, targeted_4b, kraidia_9b, bozdag_q25_7b re-run in-batch as paired anchors.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
Q4=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/* | head -1)
Q9=/path/to/models/qwen3.5-9b
Q7=/path/to/models/qwen2.5-7b-instruct
TOMAP=/path/to/models/qwen2.5-3b-instruct-tomap
ATT=$(printf '{"targeted_4b":"%s","kraidia_9b":"%s","bozdag_q25_7b":"%s","tomap_3b":"%s"}' $Q4 $Q9 $Q7 $TOMAP)
RUN=runs_local/tomap_test_9b
if [ ! -f $RUN/state.json ]; then
  $PY -m debate_ppo.eval_debate init --run $RUN --label "tomap2025_released_persuader_test_9b_rtx5090" --victim $Q9 \
    --attackers "$ATT" --conds none,targeted_4b,kraidia_9b,bozdag_q25_7b,tomap_3b \
    --seeds 3,4,5 --n 400 --answer-first --profile-set v1 --max-model-len 16384 \
    --targets lambda2_backup/runs/targets_none9b.json --tasks lambda2_backup/data/mmlupro_v1/tasks.jsonl
fi
$PY -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
$PY -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo TOMAP_LOCAL_DONE
