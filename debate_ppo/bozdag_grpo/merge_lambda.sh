#!/usr/bin/env bash
# Merge a setup-B GRPO checkpoint to HF with THEIR merge call (scripts/merge_model.sh: verl.model_merger, fsdp backend).
# Usage: merge_lambda.sh <step>   -> ~/models/bozdag-grpo-B-s<step>
set -euo pipefail
S=$1
source $HOME/venvs/verl-bozdag/bin/activate
cd $HOME/adversarial-persuasion
LOCAL_DIR=$HOME/runs/bozdag_grpo_B/checkpoints/global_step_$S/actor TARGET_DIR=$HOME/models/bozdag-grpo-B-s$S \
  ./scripts/merge_model.sh
