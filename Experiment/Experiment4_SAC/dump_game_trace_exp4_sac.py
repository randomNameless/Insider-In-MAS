#!/usr/bin/env python
"""Dump one human-auditable synchronous Experiment-4 SAC trajectory.

Run once with ``--stage before`` (SFT attacker) and once with ``--stage after``
(PPO adapter).  This is a diagnostic exporter, not the aggregate evaluator.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP1_EVAL = HERE.parent / "Experiment1_Eval"
EXP3 = HERE.parent / "Experiment3_CPWBFT"
sys.path[:0] = [str(ROOT), str(EXP1_EVAL), str(EXP3)]

from peft import PeftModel  # noqa: E402
from eval_attacks_consensus import MIXED_PERSONALITIES, build_model_rl_query  # noqa: E402
from eval_exp3_cpwbft import ConfidenceCPEnv, ModelHolder, load_causal_model  # noqa: E402
from attacker.generation import generate_attacker_batched  # noqa: E402
from sac_defense import SACBenignEngine, refine_chat, scoring_chat, self_check  # noqa: E402


def render_chat(tokenizer, chat) -> str:
    """Render exactly what the model receives, without tokenizing it."""
    try:
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("before", "after"), required=True,
                   help="before=SFT base; after=PPO adapter")
    p.add_argument("--wm_model_dir", required=True)
    p.add_argument("--llm_model_dir", required=True)
    p.add_argument("--rl_adapter", default=None)
    p.add_argument("--seed", type=int, default=817)
    p.add_argument("--max_pos", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--actor_max_message_tokens", type=int, default=48)
    p.add_argument("--wm_max_new_tokens", type=int, default=200)
    p.add_argument("--scorer_max_new_tokens", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.stage == "after" and not a.rl_adapter:
        p.error("--stage after requires --rl_adapter")
    return a


def main():
    a = parse_args()
    self_check()
    random.seed(a.seed); np.random.seed(a.seed % 2**32); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    wm, wm_tok = load_causal_model(a.wm_model_dir, a.device)
    sac = SACBenignEngine(
        wm, wm_tok, max_pos=a.max_pos, max_new_tokens=a.wm_max_new_tokens,
        scorer_max_new_tokens=a.scorer_max_new_tokens, f=1)
    holder = ModelHolder(wm, wm_tok, a.max_pos)
    actor, actor_tok = load_causal_model(a.llm_model_dir, a.device)
    if a.stage == "after":
        actor = PeftModel.from_pretrained(
            actor, str(Path(a.rl_adapter).expanduser())).eval()

    env = ConfidenceCPEnv(
        world_model=holder, max_pos=a.max_pos, max_steps=a.max_steps,
        consensus_tol=0, consensus_penalty=-10.0, reward_mode="spread",
        attacker_msg_mode="template")
    init_rng = random.Random(a.seed + a.max_steps * 1000)
    initial = [init_rng.randint(0, a.max_pos) for _ in range(4)]
    env.reset(options={"positions": initial,
                       "personalities": list(MIXED_PERSONALITIES)})
    # The legacy evaluation AttackEnv has no ``rng`` attribute, whereas the
    # shared training GameState does.  Keep a fixed local RNG in the diagnostic
    # exporter so its SimpleNamespace is compatible with the SAC interface.
    trace_rng = random.Random(a.seed)

    positions_over_time = [list(map(int, env.pos))]
    steps, rounds = [], []
    consensus = False
    for t in range(a.max_steps):
        snapshot_positions = list(map(int, env.pos))
        snapshot_messages = list(env.last_messages)
        attacker_chat = build_model_rl_query(env)
        gen = generate_attacker_batched(
            actor, actor_tok, [attacker_chat], [snapshot_positions[3]],
            max_msg_tokens=a.actor_max_message_tokens, sample=True,
            temperature=1.0, max_pos=a.max_pos)[0]

        state = SimpleNamespace(
            positions=list(snapshot_positions),
            sac_snapshot_positions=list(snapshot_positions),
            last_messages=list(snapshot_messages),
            personalities=list(env.personalities), t=int(env.t), rng=trace_rng,
            confidences=[1.0] * 4)
        benign_messages = sac.step_batched([state], sample=True)[0]
        benign_positions = list(map(int, state.positions[:3]))

        # Recreate the exact receiver-specific SAC prompts from the immutable
        # round-start snapshot.  state.positions has already received pending
        # benign actions, so never use it to build these audit prompts.
        audit_state = SimpleNamespace(
            positions=list(snapshot_positions), last_messages=list(snapshot_messages),
            personalities=list(env.personalities), t=int(env.t), rng=trace_rng)
        round_record = {
            "round_idx": t,
            "snapshot_positions": snapshot_positions,
            "snapshot_messages": snapshot_messages,
            "pending_positions": benign_positions + [int(gen.next_position)],
            "pending_messages": list(benign_messages) + [gen.message],
            "sac_receiver_scores": state.sac_scores,
            "sac_retained_neighbors": state.sac_retained_neighbors,
            "sac_filtered_neighbors": [
                [j for j in range(4) if j != i and
                 j not in state.sac_retained_neighbors[i]] for i in range(3)],
        }
        rounds.append(round_record)

        for i in range(3):
            score_prompts = {
                str(candidate): render_chat(
                    wm_tok, scoring_chat(audit_state, i, candidate, 1))
                for candidate in range(4)
            }
            refine = refine_chat(
                audit_state, i, state.sac_scores[i],
                state.sac_retained_neighbors[i], 1)
            steps.append({
                "round_idx": t,
                "agent_idx": i,
                "personality": env.personalities[i],
                "role": "benign_sac_receiver",
                "snapshot_positions_seen": snapshot_positions,
                "snapshot_messages_seen": snapshot_messages,
                "scoring_prompts": score_prompts,
                "receiver_scores": state.sac_scores[i],
                "retained_neighbors": state.sac_retained_neighbors[i],
                "prompt": render_chat(wm_tok, refine),
                "response": (f"Message: {benign_messages[i]}\n"
                             f"Position: {benign_positions[i]}"),
                "new_position": benign_positions[i],
                "broadcast_message": benign_messages[i],
                "position_dist": None,
            })
        steps.append({
            "round_idx": t,
            "agent_idx": 3,
            "personality": "malicious",
            "role": "attacker_sft_base" if a.stage == "before" else "attacker_ppo",
            "snapshot_positions_seen": snapshot_positions,
            "snapshot_messages_seen": snapshot_messages,
            "prompt": render_chat(actor_tok, attacker_chat),
            "response": f"Message: {gen.message}\nPosition: {int(gen.next_position)}",
            "new_position": int(gen.next_position),
            "broadcast_message": gen.message,
            "position_dist": None,
            "valid": bool(gen.valid),
        })

        # Atomic commit: nothing generated in round t becomes visible until here.
        env.pos[:3] = benign_positions
        env.pos[3] = int(gen.next_position)
        env.last_messages = list(benign_messages) + [gen.message]
        env.t += 1
        positions_over_time.append(list(map(int, env.pos)))
        consensus = max(env.pos[:3]) == min(env.pos[:3])
        round_record["committed_positions"] = list(map(int, env.pos))
        round_record["consensus_after"] = bool(consensus)
        if consensus:
            break

    payload = {
        "experiment": 4,
        "defense": "SAC_F1_complete4",
        "stage": a.stage,
        "round_timing": "synchronous_snapshot_atomic_commit",
        "seed": a.seed,
        "n_agents": 4,
        "consensus_scope": [0, 1, 2],
        "agent_configs": [
            {"agent_idx": i, "personality": p,
             "role": "benign" if i < 3 else "attacker"}
            for i, p in enumerate(env.personalities)],
        "initial_positions": initial,
        "positions_over_time": positions_over_time,
        "consensus": bool(consensus),
        "episode_length": len(rounds),
        "rounds": rounds,
        "steps": steps,
    }
    # Fail instead of writing a misleading trace if the timing contract ever
    # regresses.  All four actions in a round must read one identical snapshot;
    # round t+1 must start from round t's single atomic commit.
    for rr in rounds:
        same_round = [s for s in steps if s["round_idx"] == rr["round_idx"]]
        assert len(same_round) == 4
        assert all(s["snapshot_positions_seen"] == rr["snapshot_positions"]
                   for s in same_round)
        assert all(s["snapshot_messages_seen"] == rr["snapshot_messages"]
                   for s in same_round)
    for previous, current in zip(rounds, rounds[1:]):
        assert current["snapshot_positions"] == previous["committed_positions"]
        assert current["snapshot_messages"] == previous["pending_messages"]
    out = Path(a.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[trace] stage={a.stage} rounds={len(rounds)} consensus={consensus}")
    print(f"[trace] wrote {out}")


if __name__ == "__main__":
    main()
