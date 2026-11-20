#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(9000 9001 9002 9003 9004)
fi

OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-2024-08-06}"
SFT_MODEL="${SFT_MODEL:-$HOME/models/qwen3-4bFullSFTMalicious-8-24-2026}"
PPO_ADAPTER="${PPO_ADAPTER:-$HOME/checkpoints/exp1_best/training_lora/best}"
DQN_CKPT="${DQN_CKPT:-$HERE/../Experiment1_Main/runs/dqn_exp1_sync_retrain_v1/attacker_softq_sync_retrain_v1.pt}"
OUT="${OUT:-$HERE/results/exp1_sync_gpt4o_no_defense_all_conditions}"

[[ -n "${OPENAI_API_KEY:-}" ]] || { echo "[error] OPENAI_API_KEY is not set" >&2; exit 2; }
[[ -s "$SFT_MODEL/config.json" ]] || { echo "[error] missing SFT model: $SFT_MODEL" >&2; exit 2; }
[[ -s "$PPO_ADAPTER/adapter_model.safetensors" ]] || { echo "[error] missing PPO adapter: $PPO_ADAPTER" >&2; exit 2; }
[[ -s "$DQN_CKPT" ]] || { echo "[error] missing DQN checkpoint: $DQN_CKPT" >&2; exit 2; }
mkdir -p "$OUT"

for seed in "${SEEDS[@]}"; do
  json="$OUT/eval_seed${seed}.json"
  log="$OUT/eval_seed${seed}.log"
  if [[ -s "$json" ]]; then
    echo "[skip] GPT-4o seed=$seed already complete"
    continue
  fi
  echo "========== GPT-4o ALL CONDITIONS SEED $seed =========="
  python -u eval_gpt4o_no_defense.py \
    --openai_model "$OPENAI_MODEL" \
    --llm_model_dir "$SFT_MODEL" \
    --rl_adapter "$PPO_ADAPTER" \
    --dqn_ckpt "$DQN_CKPT" \
    --conditions no_attack,random,extreme,prompt,dqn,model_base,model_rl \
    --n_exp 200 --seed "$seed" --eval_batch_size 8 \
    --openai_workers 16 --openai_retries 6 \
    --openai_temperature 0.7 --openai_max_output_tokens 200 \
    --llm_temperature 0.7 --max_new_tokens 48 --log_every 0 \
    --dump_json "$json" --dump_n 200 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "[error] GPT-4o seed=$seed exit_status=$status" >&2
    exit "$status"
  fi
done
