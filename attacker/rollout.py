"""Lockstep batched rollout, matching RL_attack_env_qwen.AttackEnv dynamics.

Per round t (0-indexed), all four actions are round-synchronous:
  1. freeze the positions/messages/confidences committed at end of t-1;
  2. attacker and all benign agents generate from that same snapshot;
  3. commit all new positions/confidences/messages together at round end.
  4. Phi(benign), consensus (max-min <= tol), reward, terminal (consensus or
     t == max_steps-1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from .game import GameConfig, GameState, clamp_position, new_game
from .generation import GenTurn, generate_attacker_batched
from .potential import phi
from .prompts import ATTACKER_IDX, build_chat, build_user_prompt
from .reward import RewardConfig, compute_turn_reward


@dataclass
class EpisodeTurn:
    t: int
    prompt_text: str                 # critic state (attacker view + round info)
    gen: GenTurn
    value: float
    phi_prev: float
    phi_next: float
    valid: bool
    reward_spread: float = 0.0
    reward_terminal: float = 0.0
    reward_validity: float = 0.0
    reward_fluency: float = 0.0      # added by trainer post-recompute (needs ref_logp)
    reward_length: float = 0.0
    reward_diversity: float = 0.0
    message_tokens: int = 0
    message_hit_cap: bool = False
    message_max_similarity: float = 0.0
    msg_ref_logp: float = 0.0        # mean base log-prob of message tokens (logging)
    advantage: float = 0.0
    ret: float = 0.0
    old_logp: torch.Tensor = None
    ref_logp: torch.Tensor = None

    @property
    def reward(self) -> float:
        return (self.reward_spread + self.reward_terminal
                + self.reward_validity + self.reward_fluency
                + self.reward_length + self.reward_diversity)


@dataclass
class Episode:
    turns: List[EpisodeTurn] = field(default_factory=list)
    consensus_reached: bool = False
    length: int = 0
    phi_0: float = 0.0
    phi_T: float = 0.0


def _game_cfg(cfg) -> GameConfig:
    return GameConfig(
        max_pos=cfg.game.max_pos, n_benign=cfg.game.n_benign,
        max_steps=cfg.game.max_steps, consensus_tol=cfg.game.consensus_tol,
        benign_personality=cfg.game.benign_personality,
        benign_personalities=getattr(cfg.game, "benign_personalities", None),
        msg_order=cfg.game.msg_order,
        init=cfg.game.init, seed=cfg.game.seed,
    )


def _critic_text(state: GameState, t: int, T: int) -> str:
    # attacker's view of the state + explicit round info (§3.1 requires t/T)
    return (build_user_prompt(ATTACKER_IDX, state.positions, state.last_messages,
                              t, state.cfg.msg_order)
            + f"\nRound {t} of {T}.")


def rollout_batch(*, actor, actor_tok, critic, critic_tok, benign_engine,
                  cfg, rcfg: RewardConfig, seeds, sample: bool = True):
    gcfg = _game_cfg(cfg)
    N = len(seeds)
    states = [new_game(gcfg, seed=s) for s in seeds]
    eps = [Episode(phi_0=phi(st.benign_positions())) for st in states]
    cdev = next(critic.parameters()).device
    T = cfg.game.max_steps
    active = list(range(N))

    for t in range(T):
        if not active:
            break
        for i in active:
            states[i].t = t
        act_states = [states[i] for i in active]
        for st in act_states:
            st._round_snapshot_positions = tuple(st.positions)
            st._round_snapshot_messages = tuple(st.last_messages)
            st._round_snapshot_confidences = tuple(st.confidences)

        # ---- attacker ----
        attacker_chats = [
            build_chat(ATTACKER_IDX, "malicious", st.positions, st.last_messages,
                       t, gcfg.msg_order, st.rng) for st in act_states]
        critic_texts = [_critic_text(st, t, T) for st in act_states]
        phi_prev = [phi(st.benign_positions()) for st in act_states]
        fallbacks = [st.positions[ATTACKER_IDX] for st in act_states]

        enc = critic_tok(critic_texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            vals = critic(enc.input_ids.to(cdev), enc.attention_mask.to(cdev)).tolist()

        gts = generate_attacker_batched(
            actor, actor_tok, attacker_chats, fallbacks,
            max_msg_tokens=cfg.actor.max_new_tokens, sample=sample,
            temperature=1.0, max_pos=gcfg.max_pos)

        # Do NOT mutate state yet. Benign prompts and every defense must read the
        # same round-start snapshot that the attacker read.
        atk_msgs, atk_positions, atk_confidences = [], [], []
        for gt in gts:
            atk_msgs.append(gt.message)
            atk_positions.append(clamp_position(gt.next_position, gcfg))
            pm = gt.seg_masks["next_position"]
            atk_confidences.append(
                float(torch.exp(gt.old_logp_gen[pm].mean())) if bool(pm.any()) else 1.0)

        # ---- benign (world model) ----
        benign_msgs = benign_engine.step_batched(act_states, sample=sample)
        for st in act_states:
            assert tuple(st.last_messages) == st._round_snapshot_messages, (
                "round-synchronous violation: messages changed before commit")
            # Benign engines are allowed to commit indices 0..2 only.
            assert st.positions[ATTACKER_IDX] == st._round_snapshot_positions[ATTACKER_IDX], (
                "round-synchronous violation: attacker position visible to benign")
            assert st.confidences[ATTACKER_IDX] == st._round_snapshot_confidences[ATTACKER_IDX], (
                "round-synchronous violation: attacker confidence visible to defense")

        # ---- message lag update + reward/terminal ----
        still = []
        for idx, i in enumerate(active):
            st = states[i]
            # Atomic round commit: benign_engine has produced/committed the
            # benign actions; only now expose the attacker's new action too.
            st.positions[ATTACKER_IDX] = atk_positions[idx]
            st.confidences[ATTACKER_IDX] = atk_confidences[idx]
            st.last_messages = list(benign_msgs[idx]) + [atk_msgs[idx]]
            # public history: positions everyone read this round + what was broadcast
            st.rounds.append((list(st._round_snapshot_positions), list(st.last_messages)))
            phi_next = phi(st.benign_positions())
            consensus = st.consensus_reached()
            is_terminal = consensus or (t == T - 1)
            rew = compute_turn_reward(
                benign_prev=None, benign_next=None,
                phi_prev=phi_prev[idx], phi_next=phi_next,
                valid=gts[idx].valid, is_terminal=is_terminal,
                consensus_reached=consensus, t=t, T=T, cfg=rcfg)
            eps[i].turns.append(EpisodeTurn(
                t=t, prompt_text=critic_texts[idx], gen=gts[idx], value=vals[idx],
                phi_prev=phi_prev[idx], phi_next=phi_next, valid=gts[idx].valid,
                reward_spread=rew.spread, reward_terminal=rew.terminal,
                reward_validity=rew.validity))
            if is_terminal:
                eps[i].consensus_reached = consensus
            else:
                still.append(i)
        active = still

    for ep in eps:
        ep.length = len(ep.turns)
        ep.phi_T = ep.turns[-1].phi_next
    return eps


def assign_gae(ep: Episode, cfg) -> None:
    from .gae import gae_and_returns
    rewards = [tn.reward for tn in ep.turns]
    values = [tn.value for tn in ep.turns]
    adv, returns = gae_and_returns(
        rewards, values, gamma=cfg.gae.gamma,
        lam_actor=cfg.gae.lam_actor, lam_critic=cfg.gae.lam_critic,
        bootstrap_value=0.0)
    for tn, a, r in zip(ep.turns, adv, returns):
        tn.advantage = a
        tn.ret = r
