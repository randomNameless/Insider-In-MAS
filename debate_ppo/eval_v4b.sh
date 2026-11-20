#!/usr/bin/env bash
# Deliverables 1 (sim, WM honest) and 2 (sim2real, Qwen3.5-9B honest) for the Exp9 v4b attacker.
# One paired run per victim: none / prompt / model_base (answer-first, as trained) /
# model_base_argfirst (argument-first base, the stronger untrained baseline) / model_rl_{ckpt20,best,last}.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
R=/path/to/runs/exp9_debate_ppo_v4b_answerfirst_lr2e-5
OUT=/path/to/runs/exp9_eval_v4b
mkdir -p $OUT
for TAG in ckpt20 best last; do
  case $TAG in ckpt20) AD=$R/checkpoint-0020;; best) AD=$R/best;; last) AD=$R/interrupted-0172;; esac  # chose A at step 172: interrupted save = last
  if [ ! -f $M/exp9-v4b-rl-$TAG/config.json ]; then
    python scripts/merge_adapter.py --base $M/qwen3-4b --adapter $AD --out $M/exp9-v4b-rl-$TAG --dtype float32
  fi
done
ATT=$(printf '{"prompt":"%s","model_base":"%s","model_base_argfirst":"%s","model_rl_ckpt20":"%s","model_rl_best":"%s","model_rl_last":"%s"}' \
  $M/qwen3-4b $M/qwen3-4b $M/qwen3-4b $M/exp9-v4b-rl-ckpt20 $M/exp9-v4b-rl-best $M/exp9-v4b-rl-last)
CKPTS="ckpt20=$R/checkpoint-0020 best=$R/best($(python3 -c "import json;print(json.load(open('$R/best/meta.json'))['meta']['step'])")) last=$R/interrupted-0172"
for VIC in wm 9b; do
  if [ $VIC = wm ]; then VPATH=$M/qwen35-wm; LABEL="deliverable1_sim_WM"; else VPATH=$M/qwen3.5-9b; LABEL="deliverable2_sim2real_Qwen3.5-9B"; fi
  RUN=$OUT/$VIC
  if [ ! -f $RUN/state.json ]; then
    python -m debate_ppo.eval_debate init --run $RUN --label "$LABEL" --victim $VPATH --attackers "$ATT" \
      --rl-checkpoint "$CKPTS" --conds none,prompt,model_base,model_base_argfirst,model_rl_ckpt20,model_rl_best,model_rl_last \
      --seeds 0,1,2 --n 400 --answer-first
  fi
  python -m debate_ppo.eval_debate play --run $RUN 2>&1 | grep -vE "^\(EngineCore|^INFO|^WARNING|it/s\]" | tee -a $RUN/play.log
  python -m debate_ppo.eval_debate summary --run $RUN | tee $RUN/summary.txt
done
echo EVAL_ALL_DONE
