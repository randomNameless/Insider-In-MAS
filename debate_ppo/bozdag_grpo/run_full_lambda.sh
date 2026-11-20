#!/usr/bin/env bash
# Bozdag GRPO setup B, full 360-step run on one H100 with the persuadee on the authors' local 5090 (option A, 2026-10-25).
# Wraps run_lambda.sh train with two things their multi-GPU setup did not need:
#   * checkpoint rotation: a Qwen3-4B fp32 FSDP checkpoint (weights + Adam) is ~48 GB; their save_freq=10 over 360 steps
#     would be ~1.7 TB. verl keeps only the newest KEEP checkpoints (trainer.max_actor_ckpt_to_keep, memory/disk only,
#     no effect on training); a harvester copies the steps in KEEP_STEPS out of the rotation before they are deleted
#     (rotation happens only when the next checkpoint is written, ~1 h later), so those can be merged and evaluated.
#   * restart: if the trainer process dies (tunnel drop, vLLM hang, ...), it is relaunched up to MAX_TRIES times;
#     verl resume_mode=auto continues from the newest checkpoint. Done = latest_checkpointed_iteration.txt == STEPS.
# Lambda:  tmux new -d -s bozdag "bash ~/consensus-attacker/debate_ppo/bozdag_grpo/run_full_lambda.sh 2>&1 | tee -a ~/runs/bozdag_grpo_B/full_run.log"
set -u
CA=$HOME/consensus-attacker
OUT=$HOME/runs/bozdag_grpo_B
CKPT=$OUT/checkpoints
KEEP_DIR=$OUT/keep
STEPS=${STEPS:-360}
KEEP=${KEEP:-2}
KEEP_STEPS=${KEEP_STEPS:-"60 120 180 240 300 360"}
# Disk (461 GB free on 10-26): rotation KEEP x 47 GB + harvested copies. The harvester copies everything except the
# optimizer state (optim_*.pt, 31 of the 47 GB): verl.model_merger only needs the model shards + huggingface/ config,
# and the final checkpoint (never rotated out) still has its optimizer for a possible continuation.
MAX_TRIES=${MAX_TRIES:-4}
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
latest() { cat $CKPT/latest_checkpointed_iteration.txt 2>/dev/null || echo 0; }
mkdir -p $KEEP_DIR

harvest() {   # background: copy KEEP_STEPS checkpoints out of the rotation once verl has fully written them
  while true; do
    for s in $KEEP_STEPS; do
      src=$CKPT/global_step_$s
      if [ -d $src ] && [ ! -e $KEEP_DIR/global_step_$s/.complete ] && [ "$(latest)" -ge "$s" ]; then
        echo "[$(ts)] harvest: copying global_step_$s"
        rm -rf $KEEP_DIR/global_step_$s
        if rsync -a --exclude 'optim_*' $src/ $KEEP_DIR/global_step_$s/; then touch $KEEP_DIR/global_step_$s/.complete; echo "[$(ts)] harvest: kept global_step_$s without optimizer ($(du -sh $KEEP_DIR/global_step_$s | cut -f1)), disk free $(df -h $HOME | awk 'NR==2{print $4}')"; fi
      fi
    done
    [ "$(latest)" -ge "$STEPS" ] && [ -e $KEEP_DIR/global_step_$STEPS/.complete ] && { echo "[$(ts)] harvest: done"; return; }
    sleep 120
  done
}
harvest &
HARVEST_PID=$!

try=0
while [ "$(latest)" -lt "$STEPS" ] && [ $try -lt $MAX_TRIES ]; do
  try=$((try + 1))
  echo "[$(ts)] attempt $try/$MAX_TRIES, resuming from step $(latest)"
  STEPS=$STEPS CKPT=$CKPT bash $CA/debate_ppo/bozdag_grpo/run_lambda.sh train trainer.max_actor_ckpt_to_keep=$KEEP
  echo "[$(ts)] trainer exited ($?), latest checkpoint step $(latest)"
  [ "$(latest)" -ge "$STEPS" ] && break
  sleep 90    # let Ray / SGLang release the GPU; run_lambda.sh re-checks the tunnel before relaunching
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "[$(ts)] GPU still busy, waiting"; sleep 60; done
done
if [ "$(latest)" -ge "$STEPS" ]; then
  echo "[$(ts)] TRAINING DONE at step $(latest); waiting for the harvester"
  wait $HARVEST_PID
  echo "[$(ts)] ALL DONE"
else
  echo "[$(ts)] GAVE UP after $try attempts at step $(latest); harvester left running"
  wait $HARVEST_PID
fi
