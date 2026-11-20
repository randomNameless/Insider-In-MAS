#!/usr/bin/env bash
# Targeted PPO (v6 recipe) trained ON the strat-DPO base (2026-10-24; debate_ppo/configs/debate_v6_stratdpo_base.yaml),
# Lambda H100, honest = real Qwen3.5-9B. Same questions as every other batch: MMLU-Pro test n=400, seeds 3,4,5,
# targets_none9b.json (training used fit, WM validation val). Decoding as eval_v6_ours_lambda.sh: --answer-first (target
# letter prefilled), --arg-tokens 200, profile set v1. Checkpoints fixed IN ADVANCE (0040 / 0080 / 0120, as for v6), no
# selection on test. Each RL LoRA is merged into the strat-DPO base (fp32, as the v6 merges) -> ~/models/exp9-v6sd-tgt-c*.
#   none                      4th seat = neutral honest 9B
#   targeted_model_stratdpo   strat-DPO model, no RL LoRA = PPO step 0 (same decoder) -> isolates the PPO gain
#   targeted_model_rl_sd040/080/120   strat-DPO base + PPO LoRA at step 40 / 80 / 120
#   targeted_model_rl_c040    v6 c040 (plain Qwen3-4B base), in-batch anchor
# Started by debate_wm/chain_v6sd_lambda.sh once training has finished. SMOKE: RUN=~/runs/smoke_v6sd N=2 SEEDS=3 $0
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=/path/to/models
BASE=$M/strat-dpo-q3-4b
R=/path/to/runs/exp9_debate_ppo_v6_stratdpo_base
for k in 040 080 120; do
  OUT=$M/exp9-v6sd-tgt-c$k
  [ -f $OUT/config.json ] || python scripts/merge_adapter.py --base $BASE --adapter $R/checkpoint-0$k --out $OUT --dtype float32
done
ATT=$(printf '{"targeted_model_stratdpo":"%s","targeted_model_rl_sd040":"%s","targeted_model_rl_sd080":"%s","targeted_model_rl_sd120":"%s","targeted_model_rl_c040":"%s"}' \
  $BASE $M/exp9-v6sd-tgt-c040 $M/exp9-v6sd-tgt-c080 $M/exp9-v6sd-tgt-c120 $M/exp9-v6-tgt-c040)
RUN=${RUN:-/path/to/runs/exp9_v6sd_ours_test_9b}
if [ ! -f $RUN/state.json ]; then
  python -m debate_ppo.eval_debate init --run $RUN --label "v6_recipe_on_stratdpo_base_test_9b_h100" --victim $M/qwen3.5-9b \
    --attackers "$ATT" --conds none,targeted_model_stratdpo,targeted_model_rl_sd040,targeted_model_rl_sd080,targeted_model_rl_sd120,targeted_model_rl_c040 \
    --seeds ${SEEDS:-3,4,5} --n ${N:-400} --answer-first --profile-set v1 --arg-tokens 200 \
    --targets /path/to/runs/targets_none9b.json --tasks /path/to/data/mmlupro_v1/tasks.jsonl \
    --rl-checkpoint "sd040=$R/checkpoint-0040 sd080=$R/checkpoint-0080 sd120=$R/checkpoint-0120 c040=/path/to/runs/exp9_debate_ppo_v6_targeted/checkpoint-0040"
  for m in strat-dpo-q3-4b exp9-v6sd-tgt-c040 exp9-v6sd-tgt-c080 exp9-v6sd-tgt-c120 exp9-v6-tgt-c040 qwen3.5-9b; do
    ( cd $M/$m && sha256sum *.safetensors | sed "s#  #  $m/#" )
  done > $RUN/models_sha256.txt
fi
python -m debate_ppo.eval_debate play --run $RUN >> $RUN/play.log 2>&1
python -m debate_ppo.eval_debate summary --run $RUN > $RUN/summary.txt 2>&1
echo V6SD_EVAL_DONE
