"""
Discrete soft Q-learning (max-entropy) attacker, trained DIRECTLY against the
opd-mm Qwen world model in a MIXED-personality benign environment.

Prototype: train_attacker_maxentropy_qwen_suggestible.py. Only two things change:
  * benign combo is FIXED to 1 stubborn + 1 suggestible + 1 neutral (one of each,
    matching Experiment 1), instead of 3x suggestible;
  * the world model is the mixed/general opd-mm (one WM prompted with the three
    personality suffixes), pointed at by --model_dir; use a vLLM-loadable checkpoint.

Everything else (soft Q-learning, replay, relative-separation reward, shared WM
across envs) is identical to the suggestible trainer.

Run (on the box with the WM + vLLM):
    python train_attacker_maxentropy_qwen_mixed.py \
        --model_dir ~/models/qwen35-wm-vllm --backend vllm

The DQN's ACTION is the attacker's broadcast POSITION (0..max_pos). During DQN
training the attacker message is the fixed template "Let's all move to {pos}"
(attacker_msg_mode=template); the natural-language *rephrase* is applied only at
EVAL time (see eval_attacks_consensus.py), by a separate stock Qwen3.

Depends on RL_attack_env_qwen.py (the prototype env) being importable.
"""

import argparse
import csv
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from RL_attack_env_qwen import AttackEnv


# ---------------------------------------------------------------------------
# Mixed-personality env: 1 stubborn + 1 suggestible + 1 neutral + malicious,
# with the relative-separation reward (copied from the suggestible subclass).
# ---------------------------------------------------------------------------

MIXED_PERSONALITIES = ["stubborn", "suggestible", "neutral", "malicious"]


class AttackEnvQwenMixed(AttackEnv):
    """Qwen AttackEnv with a fixed mixed benign combo + relative-sep reward.

    The three benign agents are always {stubborn, suggestible, neutral} (one of
    each). AttackEnv.reset() canonicalizes the benign slot order by
    (personality, position), so the DQN sees a stable observation layout
    [t0,t1,t2,pos0,pos1,pos2,prev_attacker_pos] with t == (0,1,2) fixed.
    """

    def __init__(self, *args, **kwargs):
        # Force the dense reward branch off in the parent (we override apply()).
        kwargs.setdefault("reward_mode", "spread")
        super().__init__(*args, **kwargs)
        self._sep_before = 0

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        if options.get("personalities") is None:
            options["personalities"] = list(MIXED_PERSONALITIES)
        return super().reset(seed=seed, options=options)

    def prepare(self, action):
        benign = self.pos[:3]
        self._sep_before = int(benign.max() - benign.min())
        # Synchronous staging: build benign inputs from the untouched round
        # snapshot. The attacker action is committed only inside apply().
        self._sync_pending_action = int(action)
        return [(self._build_messages(i), int(self.pos[i])) for i in range(3)]

    def apply(self, preds):
        # Stage the attacker only after all benign predictions already exist.
        super().prepare(self._sync_pending_action)
        obs, _parent_reward, terminated, truncated, info = super().apply(preds)
        benign = self.pos[:3]
        sep_after = int(benign.max() - benign.min())
        sep_before = self._sep_before
        denom = max(1.0, float(sep_before))
        base_reward = (float(sep_after) - float(sep_before)) / denom
        reward = base_reward
        if terminated:
            reward += float(self.consensus_penalty)
        info = dict(info)
        info.update({"sep_before": float(sep_before), "sep_after": float(sep_after),
                     "relative_sep_change": float(base_reward)})
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Soft Q-learning (unchanged from the suggestible prototype)
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float())


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.storage: Deque[Transition] = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        self.storage.append(Transition(
            obs=np.asarray(obs, dtype=np.float32), action=int(action),
            reward=float(reward), next_obs=np.asarray(next_obs, dtype=np.float32),
            terminated=bool(terminated)))

    def sample(self, batch_size: int, device: torch.device):
        batch = random.sample(self.storage, batch_size)
        obs = torch.as_tensor(np.stack([t.obs for t in batch]), device=device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(np.stack([t.next_obs for t in batch]), device=device)
        dones = torch.as_tensor([t.terminated for t in batch], dtype=torch.float32, device=device)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self.storage)


def make_env(model_dir, backend, max_pos, max_steps, consensus_tol, consensus_penalty,
             attacker_msg_mode, max_new_tokens, world_model=None):
    return AttackEnvQwenMixed(
        model_dir=model_dir, backend=backend, max_pos=max_pos, max_steps=max_steps,
        consensus_tol=consensus_tol, consensus_penalty=consensus_penalty,
        attacker_msg_mode=attacker_msg_mode, max_new_tokens=max_new_tokens,
        world_model=world_model,
        chat_template_kwargs={"enable_thinking": False})


def soft_value(q_values: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * torch.logsumexp(q_values / alpha, dim=1)


def linear_lr_schedule(initial_lr, final_lr, step, total_steps):
    if total_steps <= 1:
        return final_lr
    progress = min(max((step - 1) / (total_steps - 1), 0.0), 1.0)
    return initial_lr + progress * (final_lr - initial_lr)


def select_action(q_net, obs, alpha, device, deterministic=False) -> int:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(obs_t).squeeze(0)
        if deterministic:
            return int(torch.argmax(q_values).item())
        probs = torch.softmax(q_values / alpha, dim=0)
        return int(torch.multinomial(probs, num_samples=1).item())


def evaluate(q_net, args, device, deterministic, eval_envs) -> float:
    rewards: List[float] = []
    for ep, env in enumerate(eval_envs):
        obs, _ = env.reset(seed=args.eval_seed + ep)
        done = False
        total_reward = 0.0
        while not done:
            action = select_action(q_net, obs, args.alpha, device, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
        rewards.append(total_reward)
    return float(np.mean(rewards))


def save_top_model(top_models, q_net, args, mean_reward, step) -> None:
    os.makedirs(args.best_path, exist_ok=True)
    should_save = len(top_models) < args.top_k or mean_reward > top_models[-1][0]
    if not should_save:
        return
    reward_tag = f"{mean_reward:.3f}".replace(".", "p")
    model_path = os.path.join(args.best_path, f"top_reward_{reward_tag}_steps_{step}.pt")
    torch.save({"model_state": q_net.state_dict(), "obs_dim": args.obs_dim,
                "n_actions": args.n_actions, "alpha": args.alpha, "args": vars(args)},
               model_path)
    top_models.append((mean_reward, model_path))
    top_models.sort(key=lambda x: x[0], reverse=True)
    while len(top_models) > args.top_k:
        _, drop_path = top_models.pop()
        if os.path.isfile(drop_path):
            os.remove(drop_path)


def main():
    parser = argparse.ArgumentParser(
        description="Soft Q-learning attacker vs. opd-mm WM, MIXED benign combo.")
    parser.add_argument("--model_dir", default=os.environ.get("WORLD_MODEL", "~/models/qwen35-wm-vllm"),
                        help="opd-mm world model dir (vLLM-loadable checkpoint).")
    parser.add_argument("--backend", default=os.environ.get("WM_BACKEND", "vllm"),
                        choices=["vllm", "hf"])
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--attacker_msg_mode", default="template", choices=["template", "rephrase"],
                        help="DQN training message channel; keep 'template' (rephrase is an EVAL-time swap).")
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--eval_freq", type=int, default=2000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--eval_seed", type=int, default=689)
    parser.add_argument("--best_path", default="best_attacker_softq_qwen_mixed")
    parser.add_argument("--eval_curve_path", default=None)
    parser.add_argument("--max_pos", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--consensus_tol", type=float, default=0.0)
    parser.add_argument("--consensus_penalty", type=float, default=-10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", default="attacker_softq_qwen_mixed.pt")
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[train] device={device} backend={args.backend} model_dir={args.model_dir}", flush=True)
    print(f"[train] benign combo = {MIXED_PERSONALITIES[:3]} (one of each)", flush=True)
    print(f"[train] loading opd-mm world model (this can take a minute)...", flush=True)

    first_env = make_env(args.model_dir, args.backend, args.max_pos, args.max_steps,
                         args.consensus_tol, args.consensus_penalty, args.attacker_msg_mode,
                         args.max_new_tokens, world_model=None)
    shared_wm = first_env.wm
    print(f"[train] world model loaded. training for {args.total_steps} steps "
          f"(learning_starts={args.learning_starts}, eval_freq={args.eval_freq}).", flush=True)

    envs = [first_env] + [
        make_env(args.model_dir, args.backend, args.max_pos, args.max_steps,
                 args.consensus_tol, args.consensus_penalty, args.attacker_msg_mode,
                 args.max_new_tokens, world_model=shared_wm)
        for _ in range(args.num_envs - 1)]
    obs_list = [env.reset(seed=args.seed + i)[0] for i, env in enumerate(envs)]

    eval_envs = [
        make_env(args.model_dir, args.backend, args.max_pos, args.max_steps,
                 args.consensus_tol, args.consensus_penalty, args.attacker_msg_mode,
                 args.max_new_tokens, world_model=shared_wm)
        for _ in range(args.n_eval_episodes)]

    args.obs_dim = int(np.asarray(obs_list[0]).shape[0])
    args.n_actions = int(envs[0].action_space.n)
    q_net = QNetwork(args.obs_dim, args.n_actions, args.hidden_dim).to(device)
    target_q_net = QNetwork(args.obs_dim, args.n_actions, args.hidden_dim).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.buffer_size)

    eval_curve_path = args.eval_curve_path or os.path.join(args.best_path, "eval_curve.csv")
    os.makedirs(os.path.dirname(eval_curve_path) or ".", exist_ok=True)
    if not os.path.isfile(eval_curve_path):
        with open(eval_curve_path, "w", newline="", encoding="utf-8") as f:
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
            with open(eval_curve_path, "a", newline="", encoding="utf-8") as f:
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
    for env in envs:
        env.close()
    for env in eval_envs:
        env.close()


if __name__ == "__main__":
    main()
