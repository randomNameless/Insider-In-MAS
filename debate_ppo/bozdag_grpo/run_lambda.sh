#!/usr/bin/env bash
# Bozdag et al. 2026 GRPO reproduction, setup B (2026-10-20), one H100 on Lambda.
#   step 1  serve the persuadee (Qwen3.5-9B, thinking off) with OUR vLLM (.venv) on :8000, GPU share 0.32, 8k context (training: prompts <= 2048 + message + 1024-token reply)
#   step 2  build the training parquet with THEIR process_task (make_data_mmlupro.py)
#   step 3  GRPO with THEIR scripts/train_grpo.sh + verl 4ed78118 (+ their patch) + sglang 0.4.9.post6
# Usage: run_lambda.sh data | train [extra verl overrides...]
# Training budget: their reported p-rl_qwen2-5-7b persuader is global_step_360 (scripts/merge_model.sh default) =
# 3 epochs x 120 steps of 24 TruthfulQA pairs. Ours has ~4x more pairs; the same 360 optimizer steps are kept via
# trainer.total_training_steps (< 1 epoch, no question repeats).
set -euo pipefail
CA=$HOME/consensus-attacker
AP=$HOME/adversarial-persuasion
OUT=$HOME/runs/bozdag_grpo_B
CFG=$CA/debate_ppo/bozdag_grpo/p-rl_qwen3-4b_persuadee-qwen3-5-9b_mmlupro.yaml
STEPS=${STEPS:-360}
mkdir -p $OUT/data
export ADVERSARIAL_PERSUASION_ROOT=$AP

serve_persuadee() {
  if curl -sf localhost:8000/v1/models >/dev/null; then return; fi
  ( source $CA/.venv/bin/activate
    VLLM_USE_FLASHINFER_SAMPLER=0 nohup vllm serve $HOME/models/qwen3.5-9b --served-model-name qwen3.5-9b --port 8000 \
      --dtype bfloat16 --gpu-memory-utilization ${PERSUADEE_GPU:-0.32} --max-model-len ${PERSUADEE_LEN:-8192} --max-num-seqs ${PERSUADEE_SEQS:-64} \
      --default-chat-template-kwargs '{"enable_thinking": false}' \
      --additional-config '{"gdn_prefill_backend": "triton"}' > $OUT/persuadee_server.log 2>&1 & )
  for i in $(seq 1 120); do
    curl -sf localhost:8000/v1/models >/dev/null && return
    grep -q "Engine core initialization failed" $OUT/persuadee_server.log 2>/dev/null && break
    sleep 10
  done
  echo "persuadee server did not come up"; exit 1
}

case "${1:-}" in
  data)
    PERSUADEE_GPU=0.85 PERSUADEE_LEN=20480 PERSUADEE_SEQS=256 serve_persuadee   # initial answers up to ~16k tokens
    source $HOME/venvs/verl-bozdag/bin/activate
    python $CA/debate_ppo/bozdag_grpo/make_data_mmlupro.py --out $OUT/data/mmlupro_fit_training_data | tee $OUT/data/make_data.log
    ;;
  train)
    shift
    # 2026-10-25 (option A): the persuadee is NOT served on this GPU any more. verl loads the fp32 actor weights,
    # gradients and Adam state (~64 GB for Qwen3-4B) onto the GPU for every update, so the 9B server (25 GB) does not
    # fit next to it (three OOMs on 10-21). It runs on the authors' local RTX 5090 (vLLM 0.23, same flags as serve_persuadee)
    # and is reached through a reverse SSH tunnel on localhost:8000 (debate_ppo/bozdag_grpo/tunnel_local.sh).
    # Refuse to start if the tunnel is down instead of silently serving the persuadee here.
    curl -sf localhost:8000/v1/models | grep -q qwen3.5-9b || { echo "persuadee not reachable on localhost:8000 (tunnel down?)"; exit 1; }
    source $HOME/venvs/verl-bozdag/bin/activate
    cd $AP
    export CONDA_DEFAULT_ENV=verl          # their launcher insists on a conda env named "verl"; this venv is that env
    export WANDB_MODE=offline WANDB_DIR=$OUT
    # val_before_train=False: verl's default runs the whole training parquet as a validation pass before step 1 (~a day);
    # their config has test_freq=-1, so no validation is part of the experiment anyway.
    ./scripts/train_grpo.sh $CFG trainer.total_training_steps=$STEPS trainer.default_local_dir=${CKPT:-$OUT/checkpoints} \
      trainer.val_before_train=False "$@"
    ;;
  *) echo "usage: $0 data|train"; exit 1 ;;
esac
