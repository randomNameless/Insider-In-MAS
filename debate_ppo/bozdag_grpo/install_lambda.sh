#!/usr/bin/env bash
# Bozdag et al. 2026 GRPO reproduction (2026-10-20): training stack on Lambda, separate from ~/consensus-attacker/.venv.
# Their README: "verl 0.5.0.dev with sglang==0.4.9.post6", verl commit not pinned (their TODO).
# First try was 4ed78118 (last commit pinning sglang 0.4.9.post6), but their scripts/train_grpo.sh passes
# actor_rollout_ref.actor.fsdp_config.model_dtype, which only exists in the PPO trainer config from d7a04699
# (2025-09-03, still 0.5.0.dev) on: their launcher fails on any earlier commit. d7a04699 is the earliest commit with
# that key, it accepts verl_persuasion.patch, and it pins torch 2.7.1 like 4ed78118. sglang stays at their stated
# 0.4.9.post6 (d7a04699's setup.py pins 0.4.10.post2), so verl is installed --no-deps on top of the 0.4.9.post6 env.
set -euo pipefail
AP=$HOME/adversarial-persuasion            # rsync of external/adversarial-persuasion (commit 191948b2)
VERL_COMMIT=d7a04699
VENV=$HOME/venvs/verl-bozdag
cd $AP
if [ ! -d verl-sglang ]; then
  git clone https://github.com/volcengine/verl.git verl-sglang
  (cd verl-sglang && git checkout $VERL_COMMIT)
  # 2026-10-26: at d7a04699 the reward manager never passes the rollout's interaction rewards to compute_score
  # (verl issue #3525); their persuasion.py reads extra_info["rollout_reward_scores"]["user_turn_rewards"], a field
  # that only exists from upstream PR #3563 (bf7aac2f, 2025-09-25) on. Without it the persuasion reward is always 0
  # and GRPO trains on format + length only (that is what happened in the first 10 steps on 10-26, run set aside as
  # ~/runs/bozdag_grpo_B/checkpoints_REWARDBUG_superseded). So their working copy was at/after bf7aac2f; cherry-pick it.
  (cd verl-sglang && git -c user.name=repro -c user.email=repro@lambda cherry-pick -x bf7aac2f)
  ./verl_patch/apply_patch.sh $AP/verl-sglang
fi
python3 -m venv $VENV
source $VENV/bin/activate
pip install -U pip wheel setuptools
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -e "$AP/verl-sglang[sglang]"
pip install "flash-attn==2.8.3" --no-build-isolation || pip install \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install "torch-memory-saver>=0.0.5"     # sglang rollout sleep/wake; not pulled in by the --no-deps verl install
pip install "uvloop==0.21.0"                 # uvloop>=0.22 raises in asyncio.get_event_loop() without a loop (verl fsdp_workers init)
pip install -r $AP/requirements.txt
python - <<'EOF'
import torch, sglang, verl, flash_attn, transformers
print("torch", torch.__version__, "sglang", sglang.__version__, "verl", verl.__version__,
      "flash_attn", flash_attn.__version__, "transformers", transformers.__version__, "cuda", torch.cuda.is_available())
EOF
echo INSTALL_DONE
