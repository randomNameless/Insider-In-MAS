#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(9000 9001 9002 9003 9004)
fi

SFT="${SFT:-/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft}"
LORA="${LORA:-/path/to/consensus-attacker/protected_checkpoints/exp1_ppo_v4_best_2026-09-04}"
DQN="${DQN:-/path/to/consensus-attacker/Experiment/Experiment1_Main/runs/dqn_exp1_sync_retrain_v1/attacker_softq_sync_retrain_v1.pt}"
ROOT="${ROOT:-$HERE/results/exp1_sync_transfer_large_models_all}"
N_EXP="${N_EXP:-200}"
DUMP_N="${DUMP_N:-200}"

# tag|local model|vLLM utilisation|game batch
SPECS=(
  "qwen25_14b|$HOME/models/Qwen2.5-14B-Instruct|0.55|16"
  "mistral_small_24b_2501|$HOME/models/Mistral-Small-24B-Instruct-2501|0.65|12"
  "qwen25_32b|$HOME/models/Qwen2.5-32B-Instruct|0.78|8"
)

for required in "$SFT/config.json" "$LORA/adapter_model.safetensors" "$DQN"; do
  if [[ ! -s "$required" ]]; then
    echo "[error] required file missing: $required" >&2
    exit 1
  fi
done

for spec in "${SPECS[@]}"; do
  IFS='|' read -r tag model mem batch <<< "$spec"
  if [[ ! -s "$model/config.json" ]]; then
    echo "[error] model missing or incomplete: $model" >&2
    exit 1
  fi
  for seed in "${SEEDS[@]}"; do
    no_dir="$ROOT/$tag/no_attack"
    atk_dir="$ROOT/$tag/attacks"
    mkdir -p "$no_dir" "$atk_dir"

    no_json="$no_dir/eval_seed${seed}.json"
    if [[ -s "$no_json" ]]; then
      echo "[skip] $tag no_attack seed=$seed"
    else
      echo "========== $tag NO ATTACK seed=$seed =========="
      python -u eval_no_attacker_consensus.py \
        --wm_model_dir "$model" --wm_backend vllm \
        --n_exp "$N_EXP" --seed "$seed" --eval_batch_size "$batch" \
        --wm_gpu_mem_util "$mem" --max_model_len 2048 \
        --wm_max_new_tokens 200 --log_every 50 \
        --dump_json "$no_json" --dump_n "$DUMP_N" \
        2>&1 | tee "$no_dir/eval_seed${seed}.log"
      status=${PIPESTATUS[0]}
      [[ $status -eq 0 ]] || exit "$status"
    fi

    atk_json="$atk_dir/eval_seed${seed}.json"
    if [[ -s "$atk_json" ]]; then
      echo "[skip] $tag attacks seed=$seed"
    else
      echo "========== $tag SIX ATTACKERS seed=$seed =========="
      python -u eval_attacks_consensus.py \
        --wm_model_dir "$model" --wm_backend vllm \
        --llm_model_dir "$SFT" --rl_adapter "$LORA" \
        --dqn_ckpt "$DQN" \
        --attacks random,extreme,prompt,dqn,model_base,model_rl \
        --n_exp "$N_EXP" --seed "$seed" --eval_batch_size "$batch" \
        --wm_gpu_mem_util "$mem" --max_model_len 2048 \
        --wm_max_new_tokens 200 --max_new_tokens 48 --log_every 50 \
        --dump_json "$atk_json" --dump_n "$DUMP_N" \
        2>&1 | tee "$atk_dir/eval_seed${seed}.log"
      status=${PIPESTATUS[0]}
      [[ $status -eq 0 ]] || exit "$status"
    fi
  done
done

python "$HERE/summarize_transfer_large_models.py" \
  --result_root "$ROOT" --seeds "${SEEDS[@]}"
