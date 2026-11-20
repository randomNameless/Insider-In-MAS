#!/usr/bin/env bash
# After PPO v5 (honest = distilled debate WM):
#  1) merge every checkpoint-XXXX LoRA onto Qwen3-4B,
#  2) SELECTION on val (160 q x seeds 0,1) with the REAL 9B honest agents: none / model_base / every checkpoint,
#     pick the checkpoint with the largest paired drop in honest final accuracy vs none,
#  3) TEST (400 q x seeds 0,1,2) with honest = new WM (sim) and = 9B (sim2real):
#     none / prompt / model_base / old-WM PPO step20 (v4b) / new-WM PPO selected (v5).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
R=/path/to/runs/exp9_debate_ppo_v5_newwm
until grep -q "\[done\]" /path/to/runs/exp9_debate_ppo_v5_newwm.log; do sleep 60; done
tmux kill-session -t watchdog 2>/dev/null || true
sleep 30

CKS=$(ls -d $R/checkpoint-* | sort)
ATT='{"model_base":"'$M/qwen3-4b'"'
CONDS="none,model_base"
for CK in $CKS; do
  S=${CK##*-}; OUT=$M/exp9-v5-rl-c$S
  [ -f $OUT/config.json ] || python scripts/merge_adapter.py --base $M/qwen3-4b --adapter $CK --out $OUT --dtype float32
  ATT="$ATT,\"model_rl_c$S\":\"$OUT\""; CONDS="$CONDS,model_rl_c$S"
done
ATT="$ATT}"

SEL=/path/to/runs/exp9_v5_select_val9b
if [ ! -f $SEL/state.json ]; then
  python -m debate_ppo.eval_debate init --run $SEL --label "v5_checkpoint_selection_val_9b" --split val --n 160 \
    --victim $M/qwen3.5-9b --attackers "$ATT" --conds $CONDS --seeds 0,1 --answer-first --profile-set v1 \
    --rl-checkpoint "$R/checkpoint-*"
fi
python -m debate_ppo.eval_debate play --run $SEL > $SEL/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $SEL > $SEL/summary.txt 2>&1
BEST=$(python - <<EOF
import json
r = json.load(open("$SEL/results.json"))
c = {k[:-len(" - none")]: v["final_acc"]["diff"] for k, v in r["paired"].items()
     if k.startswith("model_rl_c") and k.endswith(" - none")}
best = min(c, key=c.get)
json.dump({"by_ckpt_diff_vs_none": c, "best": best}, open("$SEL/selection.json", "w"), indent=1)
print(best)
EOF
)
echo "selected $BEST"
BEST_PATH=$M/exp9-v5-rl-${BEST#model_rl_}

ATT2=$(printf '{"prompt":"%s","model_base":"%s","model_rl_oldwm_ckpt20":"%s","model_rl_newwm_sel":"%s"}' \
  $M/qwen3-4b $M/qwen3-4b $M/exp9-v4b-rl-ckpt20 $BEST_PATH)
for VIC in newwm 9b; do
  if [ $VIC = newwm ]; then VP=$M/debate-wm-v1; else VP=$M/qwen3.5-9b; fi
  RUN=/path/to/runs/exp9_v5_test_$VIC
  if [ ! -f $RUN/state.json ]; then
    python -m debate_ppo.eval_debate init --run $RUN --label "v5_test_honest_$VIC" --victim $VP --attackers "$ATT2" \
      --conds none,prompt,model_base,model_rl_oldwm_ckpt20,model_rl_newwm_sel --seeds 0,1,2 --n 400 \
      --answer-first --profile-set v1 --rl-checkpoint "oldwm=v4b/checkpoint-0020 newwm=$BEST ($BEST_PATH)"
  fi
  python -m debate_ppo.eval_debate play --run $RUN > $RUN/play.log 2>&1
  python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
done
echo AFTER_V5_DONE
