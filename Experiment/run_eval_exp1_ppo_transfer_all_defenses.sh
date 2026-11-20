#!/usr/bin/env bash
# Evaluate the Experiment-1 no-defense PPO attacker under every implemented
# defense, on qwen35-wm and Qwen3.5-9B. Existing non-empty JSON files are skipped.
set -uo pipefail

REPO="${REPO:-$HOME/consensus-attacker}"
BASE="${BASE:-/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft}"
EXP1_ADAPTER="${EXP1_ADAPTER:-$REPO/Experiment/Experiment1_Main/runs/exp1_nodefense_qwen3_ppo_dominant_v4/best}"
RESULT_ROOT="${RESULT_ROOT:-$REPO/Experiment/results/exp1_ppo_transfer_to_defenses}"
N_EXP="${N_EXP:-200}"
DUMP_N="${DUMP_N:-$N_EXP}"

if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(9000 9001 9002 9003 9004)
fi

MODELS=(
  "wm:$HOME/models/qwen35-wm"
  "qwen35_9b:$HOME/models/qwen3.5-9b"
)

[[ -s "$BASE/config.json" ]] || { echo "[error] missing SFT base: $BASE" >&2; exit 2; }
[[ -s "$EXP1_ADAPTER/adapter_model.safetensors" ]] || {
  echo "[error] missing Experiment-1 PPO adapter: $EXP1_ADAPTER" >&2; exit 2;
}
mkdir -p "$RESULT_ROOT"

run_logged() {
  local json="$1" log="$2"
  shift 2
  if [[ -s "$json" ]]; then
    echo "[skip] already complete: $json"
    return 0
  fi
  mkdir -p "$(dirname "$json")"
  "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  if [[ $status -ne 0 ]]; then
    echo "[error] exit_status=$status; see $log" >&2
    exit "$status"
  fi
  [[ -s "$json" ]] || { echo "[error] command returned without JSON: $json" >&2; exit 3; }
}

for spec in "${MODELS[@]}"; do
  tag="${spec%%:*}"
  model="${spec#*:}"
  [[ -s "$model/config.json" ]] || { echo "[error] missing victim model: $model" >&2; exit 2; }

  for seed in "${SEEDS[@]}"; do
    echo "========== EXP2 W-MSR | $tag | seed=$seed =========="
    out="$RESULT_ROOT/exp2_wmsr/$tag"
    run_logged "$out/eval_seed${seed}.json" "$out/eval_seed${seed}.log" \
      python -u "$REPO/Experiment/Experiment2_WMSR/eval_exp2_wmsr.py" \
        --wmsr_F 1 --wm_model_dir "$model" --wm_backend vllm \
        --llm_model_dir "$BASE" --rl_adapter "$EXP1_ADAPTER" \
        --attacks model_rl --n_exp "$N_EXP" --seed "$seed" \
        --eval_batch_size 16 --wm_gpu_mem_util 0.60 --max_model_len 2048 \
        --max_new_tokens 48 --wm_max_new_tokens 200 --log_every 0 \
        --dump_json "$out/eval_seed${seed}.json" --dump_n "$DUMP_N"

    echo "========== EXP3 CP-WBFT | $tag | seed=$seed =========="
    out="$RESULT_ROOT/exp3_cpwbft/$tag"
    run_logged "$out/eval_seed${seed}.json" "$out/eval_seed${seed}.log" \
      python -u "$REPO/Experiment/Experiment3_CPWBFT/eval_exp3_cpwbft.py" \
        --wm_model_dir "$model" --llm_model_dir "$BASE" \
        --rl_adapter "$EXP1_ADAPTER" --attacks model_rl \
        --n_exp "$N_EXP" --seed "$seed" --eval_batch_size 8 \
        --actor_max_message_tokens 48 --wm_max_new_tokens 200 --margin 0 \
        --dump_json "$out/eval_seed${seed}.json" --dump_n "$DUMP_N"

    echo "========== EXP4 SAC | $tag | seed=$seed =========="
    out="$RESULT_ROOT/exp4_sac/$tag"
    run_logged "$out/eval_seed${seed}.json" "$out/eval_seed${seed}.log" \
      python -u "$REPO/Experiment/Experiment4_SAC/eval_exp4_sac.py" \
        --wm_model_dir "$model" --wm_backend vllm \
        --wm_gpu_mem_util 0.60 --max_model_len 2048 \
        --llm_model_dir "$BASE" --rl_adapter "$EXP1_ADAPTER" \
        --attacks model_rl --n_exp "$N_EXP" --seed "$seed" \
        --eval_batch_size 16 --actor_max_message_tokens 48 \
        --wm_max_new_tokens 200 --log_every 0 \
        --dump_json "$out/eval_seed${seed}.json" --dump_n "$DUMP_N"
  done
done

echo "[complete] Experiment-1 PPO transfer evaluation: $RESULT_ROOT"
