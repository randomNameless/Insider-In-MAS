#!/usr/bin/env bash
# Deliverables 1 (sim, WM honest agents) and 2 (sim2real, Qwen3.5-9B honest agents) on test 400 x 3 seeds.
#   bash debate_ppo/eval_all.sh <adapter_dir> <tag>
# e.g. bash debate_ppo/eval_all.sh /path/to/runs/exp9_debate_ppo_v1/best best
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

ADAPTER=${1:?adapter dir}
TAG=${2:?tag}
M=/path/to/models
MERGED=$M/exp9-rl-merged-$TAG
OUT=/path/to/runs/exp9_eval

if [ ! -f "$MERGED/config.json" ]; then
  python scripts/merge_adapter.py --base ${ACTOR_BASE:-$M/qwen3-4b} --adapter "$ADAPTER" --out "$MERGED" --dtype float32
fi
ATT=$(printf '{"prompt":"%s","model_base":"%s","model_rl":"%s"}' "$M/qwen3-4b" "${ACTOR_BASE:-$M/qwen3-4b}" "$MERGED")

for VIC in wm 9b; do
  if [ $VIC = wm ]; then VPATH=$M/qwen35-wm; LABEL="deliverable1_sim_WM"; else VPATH=$M/qwen3.5-9b; LABEL="deliverable2_sim2real_Qwen3.5-9B"; fi
  RUN=$OUT/${TAG}_$VIC
  if [ ! -f $RUN/state.json ]; then
    python -m debate_ppo.eval_debate init --run $RUN --label "$LABEL" --victim $VPATH \
      --attackers "$ATT" --rl-checkpoint "$ADAPTER" --seeds 0,1,2 --n 400 ${ANSWER_FIRST:+--answer-first}
  fi
  python -m debate_ppo.eval_debate play --run $RUN 2>&1 | grep -vE "^\(EngineCore|^INFO|^WARNING|it/s\]" | tee -a $RUN/play.log
  python -m debate_ppo.eval_debate summary --run $RUN | tee $RUN/summary.txt
done
