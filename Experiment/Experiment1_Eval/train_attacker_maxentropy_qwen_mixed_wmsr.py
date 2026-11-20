"""
Experiment 2 — Discrete soft Q-learning (max-entropy) attacker vs the opd-mm WM
in the MIXED-personality environment, WITH the benign agents running a W-MSR input
filter (the same defense the PPO attacker trains against in exp2_W-MSR.py).

Based on train_attacker_maxentropy_qwen_mixed.py. The ONLY change is the env: the
benign (victim) agents drop the F most extreme neighbor positions relative to their
own (position + message) from their prompt before the WM generates. So the DQN
attacker must keep its broadcast position inside the retained band to be seen — it
learns to attack against the defense, not the naive WM.

All soft-Q machinery (QNetwork / replay / soft value / eval / checkpointing) is
reused from train_attacker_maxentropy_qwen_mixed.py.

Artifacts are written under ./W-MSR/ .

Run (on the box with the WM + vLLM):
    python train_attacker_maxentropy_qwen_mixed_wmsr.py \
        --model_dir ~/models/qwen35-wm-vllm --backend vllm --wmsr_F 1

Depends on RL_attack_env_qwen.py AND train_attacker_maxentropy_qwen_mixed.py being
importable (same directory).
"""

import argparse
import csv
import os
import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL_attack_env_qwen import (
    build_system_prompt, ROUND0_PROMPT_TEMPLATE, MULTIROUND_PROMPT_TEMPLATE,
)
# reuse the mixed trainer's env + all soft-Q machinery (same directory)
from train_attacker_maxentropy_qwen_mixed import (
    AttackEnvQwenMixed, MIXED_PERSONALITIES, QNetwork, ReplayBuffer,
    soft_value, linear_lr_schedule, select_action, evaluate, save_top_model,
)


# ---------------------------------------------------------------------------
# Mixed env + W-MSR benign input filter (mixed personalities + relative-sep
# reward come from AttackEnvQwenMixed; we only add the filter on benign prompts).
# ---------------------------------------------------------------------------

class AttackEnvQwenMixedWMSR(AttackEnvQwenMixed):
    def __init__(self, *args, wmsr_F=1, **kwargs):
        self.wmsr_F = int(wmsr_F)
        super().__init__(*args, **kwargs)

    def _visible_others(self, i):
        x = self.pos[i]
        others = [j for j in range(4) if j != i]
        above = sorted((j for j in others if self.pos[j] > x), key=lambda j: self.pos[j], reverse=True)
        below = sorted((j for j in others if self.pos[j] < x), key=lambda j: self.pos[j])
        trimmed = set(above[:self.wmsr_F]) | set(below[:self.wmsr_F])
        return [j for j in others if j not in trimmed]

    def _build_messages(self, i):
        """Benign i sees only its W-MSR-retained neighbors (position + message)."""
        vis = set(self._visible_others(i))
        others = [j for j in range(4) if j != i and j in vis]
        others_pos = [int(self.pos[j]) for j in others]
        if self.t == 0:
            user = ROUND0_PROMPT_TEMPLATE.format(
                self_pos=int(self.pos[i]), other_positions=others_pos)
        else:
            order = [j for j in self._order_others([j for j in range(4) if j != i]) if j in vis]
            lines = "\n".join(f"- Agent {j}: {self.last_messages[j]}" for j in order)
            user = MULTIROUND_PROMPT_TEMPLATE.format(
                self_pos=int(self.pos[i]), other_positions=others_pos, message_lines=lines)
        return [
            {"role": "system", "content": build_system_prompt(self.personalities[i])},
            {"role": "user", "content": user},
        ]


def make_env(model_dir, backend, max_pos, max_steps, consensus_tol, consensus_penalty,
             attacker_msg_mode, max_new_tokens, wmsr_F, world_model=None):
    return AttackEnvQwenMixedWMSR(
        model_dir=model_dir, backend=backend, max_pos=max_pos, max_steps=max_steps,
        consensus_tol=consensus_tol, consensus_penalty=consensus_penalty,
        attacker_msg_mode=attacker_msg_mode, max_new_tokens=max_new_tokens,
        world_model=world_model, chat_template_kwargs={"enable_thinking": False},
        wmsr_F=wmsr_F)


def main():
    parser = argparse.ArgumentParser(
        description="Soft Q-learning attacker vs. opd-mm WM WITH W-MSR filter, MIXED benign combo.")
    parser.add_argument("--model_dir", default=os.environ.get("WORLD_MODEL", "~/models/qwen35-wm-vllm"),
                        help="opd-mm world model dir (vLLM-loadable checkpoint).")
    parser.add_argument("--backend", default=os.environ.get("WM_BACKEND", "vllm"), choices=["vllm", "hf"])
    parser.add_argument("--wmsr_F", type=int, default=1, help="W-MSR trim count (defense strength).")
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--attacker_msg_mode", default="template", choices=["template", "rephrase"],
                        help="DQN training message channel; keep 'template' (rephrase is an EVAL-time swap).")
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--eval_freq", type=int, default=2000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--eval_seed", type=int, default=689)
    # ---- all artifacts under ./W-MSR/ ----
    parser.add_argument("--out_dir", default="W-MSR", help="output dir (created under ./).")
    parser.add_argument("--best_path", default=None, help="default: <out_dir>/best")
    parser.add_argument("--save_path", default=None, help="default: <out_dir>/attacker_softq_wmsr.pt")
    parser.add_argument("--eval_curve_path", default=None, help="default: <out_dir>/eval_curve.csv")
    parser.add_argument("--max_pos", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--consensus_tol", type=float, default=0.0)
    parser.add_argument("--consensus_penalty", type=float, default=-10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--final_learning_rate", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--log_freq", type=int, default=50)
    parser.add_argument("--eval_stochastic", action="store_true")
    args = parser.parse_args()

    # resolve output paths under ./W-MSR/
    os.makedirs(args.out_dir, exist_ok=True)
    if args.best_path is None:
        args.best_path = os.path.join(args.out_dir, "best")
    if args.save_path is None:
        args.save_path = os.path.join(args.out_dir, "attacker_softq_wmsr.pt")
    if args.eval_curve_path is None:
        args.eval_curve_path = os.path.join(args.out_dir, "eval_curve.csv")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[train] device={device} backend={args.backend} model_dir={args.model_dir}", flush=True)
    print(f"[train] benign combo = {MIXED_PERSONALITIES[:3]} (one of each)", flush=True)
    print(f"[train] DEFENSE = W-MSR benign input filter, F={args.wmsr_F}", flush=True)
    print(f"[train] artifacts -> {args.out_dir}/", flush=True)
    print(f"[train] loading opd-mm world model (this can take a minute)...", flush=True)

    first_env = make_env(args.model_dir, args.backend, args.max_pos, args.max_steps,
                         args.consensus_tol, args.consensus_penalty, args.attacker_msg_mode,
                         args.max_new_tokens, args.wmsr_F, world_model=None)
    shared_wm = first_env.wm
    print(f"[train] world model loaded. training for {args.total_steps} steps "
          f"(learning_starts={args.learning_starts}, eval_freq={args.eval_freq}).", flush=True)

    envs = [first_env] + [
        make_env(args.model_dir, args.backend, args.max_pos, args.max_steps,
                 args.consensus_tol, args.consensus_penalty, args.attacker_msg_mode,
                 args.max_new_tokens, args.wmsr_F, world_model=shared_wm)
        for _ in range(args.num_envs - 1)]
    obs_list = [env.reset(seed=args.seed + i)[0] for i, env in enumerate(envs)]

    eval_envs = [
        make_env(args.model_dir, args.backend, args.max_pos, args.max_steps,
                 args.consensus_tol, args.consensus_penalty, args.attacker_msg_mode,
                 args.max_new_tokens, args.wmsr_F, world_model=shared_wm)
        for _ in range(args.n_eval_episodes)]

    args.obs_dim = int(np.asarray(obs_list[0]).shape[0])
    args.n_actions = int(envs[0].action_space.n)
    q_net = QNetwork(args.obs_dim, args.n_actions, args.hidden_dim).to(device)
    target_q_net = QNetwork(args.obs_dim, args.n_actions, args.hidden_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.buffer_size)

    os.makedirs(os.path.dirname(args.eval_curve_path) or ".", exist_ok=True)
    if not os.path.isfile(args.eval_curve_path):
        with open(args.eval_curve_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timesteps", "mean_reward"])

    top_models: List[Tuple[float, str]] = []
    open_episode_rewards = [0.0 for _ in envs]
    completed_episode_rewards: Deque[float] = deque(maxlen=100)
    train_step = 0

    for step in range(1, args.total_steps + 1):
        env_idx = (step - 1) % args.num_envs
        env = envs[env_idx]
        obs = obs_list[env_idx]
        action = select_action(q_net, obs, args.alpha, device, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        replay.add(obs, action, reward, next_obs, terminated)
        open_episode_rewards[env_idx] += float(reward)
        if terminated or truncated:
            completed_episode_rewards.append(open_episode_rewards[env_idx])
            open_episode_rewards[env_idx] = 0.0
            obs_list[env_idx] = env.reset()[0]
        else:
            obs_list[env_idx] = next_obs

        if len(replay) >= args.learning_starts:
            train_step += 1
            current_lr = linear_lr_schedule(args.learning_rate, args.final_learning_rate,
                                            train_step, args.total_steps - args.learning_starts)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            for _ in range(args.gradient_steps):
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = replay.sample(args.batch_size, device)
                q_values = q_net(b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_v = soft_value(target_q_net(b_next_obs), args.alpha)
                    target = b_rewards + args.gamma * (1.0 - b_dones) * next_v
                loss = F.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
                optimizer.step()

        if step % args.target_update_interval == 0:
            with torch.no_grad():
                for tp, p in zip(target_q_net.parameters(), q_net.parameters()):
                    tp.data.mul_(1.0 - args.tau).add_(p.data, alpha=args.tau)

        if args.log_freq > 0 and step % args.log_freq == 0:
            print(f"[heartbeat] step={step}/{args.total_steps} replay={len(replay)} "
                  f"wm_queries={shared_wm.n_queries} wm_cache_hits={shared_wm.n_cache_hits} "
                  f"last_reward={reward:.3f}", flush=True)

        if step % args.eval_freq == 0:
            mean_reward = evaluate(q_net, args, device, not args.eval_stochastic, eval_envs)
            with open(args.eval_curve_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([step, f"{mean_reward:.6f}"])
            save_top_model(top_models, q_net, args, mean_reward, step)
            recent = float(np.mean(completed_episode_rewards)) if completed_episode_rewards else float("nan")
            print(f"step={step} mean_eval_reward={mean_reward:.4f} "
                  f"mean_recent_episode_reward={recent:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.6g} replay={len(replay)} "
                  f"wm_queries={shared_wm.n_queries} wm_cache_hits={shared_wm.n_cache_hits}", flush=True)

    torch.save({"model_state": q_net.state_dict(), "obs_dim": args.obs_dim,
                "n_actions": args.n_actions, "alpha": args.alpha, "args": vars(args)},
               args.save_path)
    print(f"[train] saved final -> {args.save_path}", flush=True)
    for env in envs:
        env.close()
    for env in eval_envs:
        env.close()


if __name__ == "__main__":
    main()
