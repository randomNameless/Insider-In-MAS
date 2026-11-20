#!/usr/bin/env bash
# Literature baselines, inference-time arms (2026-10-20), run on the local RTX 5090; honest = real Qwen3.5-9B.
# Same split (test, n=400) / seeds (3,4,5) / targets (targets_none9b.json) as exp9_v6_targeted_test_9b on Lambda;
# none + targeted_4b re-run in-batch as references (different GPU -> pair only within this run).
#   amayuelas_*     Amayuelas 2024 prompt adversary (App. E), 1000-token cap (supersedes the 200-token Lambda run)
#   amayuelasbon_*  + Best-of-10 argument selection, judge dummy = gold (App. C)            debate_ppo/baseline_bon.py
#   kraidia_*       Kraidia 2026 PPA                                                         debate_ppo/baseline_kraidia.py
#   kraidiabon_*    + BoN-A (AEG + PAS), judge dummy = target (their code)                   debate_ppo/baseline_bon.py
#   bozdag_q25_7b   Bozdag 2026 static-prompt persuader with their persuader model Qwen2.5-7B-Instruct
#                   (Qwen3-4B / Qwen3.5-9B produce 0% <message> format; runs_local/smoke_prompt, smoke2)
# Not here: Kraidia CTX (TruthfulQA-only context), Bozdag GRPO (Lambda).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
Q4=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/* | head -1)
Q9=/path/to/models/qwen3.5-9b
Q7=/path/to/models/qwen2.5-7b-instruct
ATT=$(printf '{"targeted_4b":"%s","amayuelas_4b":"%s","amayuelas_9b":"%s","amayuelasbon_4b":"%s","amayuelasbon_9b":"%s","kraidia_4b":"%s","kraidia_9b":"%s","kraidiabon_4b":"%s","kraidiabon_9b":"%s","bozdag_q25_7b":"%s"}' \
  $Q4 $Q4 $Q9 $Q4 $Q9 $Q4 $Q9 $Q4 $Q9 $Q7)
RUN=runs_local/baselines_test_9b
if [ ! -f $RUN/state.json ]; then
  $PY -m debate_ppo.eval_debate init --run $RUN --label "literature_baselines_inference_test_9b_rtx5090" --victim $Q9 \
    --attackers "$ATT" \
    --conds none,targeted_4b,amayuelas_4b,amayuelas_9b,amayuelasbon_4b,amayuelasbon_9b,kraidia_4b,kraidia_9b,kraidiabon_4b,kraidiabon_9b,bozdag_q25_7b \
    --seeds 3,4,5 --n 400 --answer-first --profile-set v1 --max-model-len 16384 \
    --targets lambda2_backup/runs/targets_none9b.json --tasks lambda2_backup/data/mmlupro_v1/tasks.jsonl
fi
$PY -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
$PY -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo BASELINES_LOCAL_DONE
