#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/consensus-attacker"
VENV="$ROOT/.venv-vllm/bin/activate"
EXP4="$ROOT/Experiment/Experiment4_SAC"
EXP1E="$ROOT/Experiment/Experiment1_Eval"
LLAMA_REPO="meta-llama/Meta-Llama-3.1-8B-Instruct"
LLAMA_DIR="$HOME/models/Meta-Llama-3.1-8B-Instruct"
BASE="/path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft"
ADAPTER="${ADAPTER:-$ROOT/Experiment/Experiment1_Main/runs/exp1_nodefense_qwen3_ppo_dominant_v4/best}"
SEEDS=(9000 9001 9002 9003 9004)

source "$VENV"
echo "[overnight] start $(date -Is)"
python --version

test -s "$BASE/model.safetensors" || { echo "missing SFT base: $BASE" >&2; exit 1; }
test -s "$ADAPTER/adapter_model.safetensors" || { echo "missing Exp1 no-defense PPO adapter: $ADAPTER" >&2; exit 1; }

llama_complete() {
  [[ -s "$LLAMA_DIR/config.json" ]] &&
    { [[ -s "$LLAMA_DIR/model.safetensors" ]] ||
      [[ -s "$LLAMA_DIR/model.safetensors.index.json" ]]; }
}

echo "[preflight] Download/check Meta-Llama-3.1-8B-Instruct"
if ! llama_complete; then
  command -v hf >/dev/null || {
    echo "missing 'hf' CLI; install huggingface_hub in .venv-vllm" >&2
    exit 1
  }
  hf auth whoami
  mkdir -p "$LLAMA_DIR"
  hf download "$LLAMA_REPO" --local-dir "$LLAMA_DIR"
fi
llama_complete || {
  echo "Llama download incomplete: config plus safetensors weights are required" >&2
  exit 1
}

echo "[1/2] SAC no-attacker: qwen35-wm and qwen3.5-9b"
cd "$EXP4"
chmod +x run_eval_exp4_no_attacker.sh
./run_eval_exp4_no_attacker.sh

echo "[2/2] Experiment 1 no-defense transfer evaluation: Meta-Llama-3.1-8B-Instruct"
cd "$EXP1E"
OUT="results/exp1_sync_transfer_meta_llama31_8b_no_defense"
mkdir -p "$OUT"
for SEED in "${SEEDS[@]}"; do
  JSON="$OUT/eval_seed${SEED}.json"
  if [[ -s "$JSON" ]] && python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["summary"]["model_rl"]["episodes"] == 200' "$JSON" 2>/dev/null; then
    echo "[skip] completed $JSON"
    continue
  fi
  python -u eval_attacks_consensus.py \
    --wm_model_dir "$LLAMA_DIR" --wm_backend vllm \
    --llm_model_dir "$BASE" --rl_adapter "$ADAPTER" \
    --attacks model_rl \
    --n_exp 200 --seed "$SEED" --eval_batch_size 16 \
    --max_new_tokens 48 --wm_max_new_tokens 200 \
    --wm_gpu_mem_util 0.60 --log_every 0 \
    --dump_json "$JSON" --dump_n 200 \
    2>&1 | tee "$OUT/eval_seed${SEED}.log"
done

echo "[overnight] COMPLETE $(date -Is)"
