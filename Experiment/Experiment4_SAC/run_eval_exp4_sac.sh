#!/usr/bin/env bash
set -euo pipefail
TARGET="${1:?usage: $0 wm|qwen35-9b [seeds...]}"; shift
if (( $# )); then SEEDS=("$@"); else SEEDS=(9000 9001 9002 9003 9004); fi
ROOT="$HOME/consensus-attacker"
case "$TARGET" in
  wm) WM="$HOME/models/qwen35-wm"; TAG="wm" ;;
  qwen35-9b) WM="$HOME/models/qwen3.5-9b"; TAG="qwen35_9b_transfer" ;;
  *) echo "target must be wm or qwen35-9b" >&2; exit 2 ;;
esac
OUT="results/exp4_sync_sac_v2_all_attackers_${TAG}"; mkdir -p "$OUT"
DQN="${DQN:-$HOME/consensus-attacker/Experiment/Experiment1_Main/runs/dqn_exp1_sync_retrain_v1/attacker_softq_sync_retrain_v1.pt}"
test -s "$DQN" || { echo "missing DQN checkpoint: $DQN" >&2; exit 1; }
for SEED in "${SEEDS[@]}"; do
  python -u eval_exp4_sac.py \
    --wm_model_dir "$WM" \
    --wm_backend vllm --wm_gpu_mem_util 0.65 \
    --llm_model_dir /path/to/LLaMA-Factory/saves/qwen3-4b/malicious-full-sft \
    --rl_adapter runs/exp4_sync_sac_qwen3_ppo_v4/best \
    --dqn_ckpt "$DQN" \
    --attacks random,extreme,prompt,dqn,model_base,model_rl \
    --n_exp 200 --eval_batch_size 16 --seed "$SEED" --log_every 50 \
    --dump_json "$OUT/eval_seed${SEED}.json" --dump_n 200 \
    2>&1 | tee "$OUT/eval_seed${SEED}.log"
done
