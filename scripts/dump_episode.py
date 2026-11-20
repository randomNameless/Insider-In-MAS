#!/usr/bin/env python
"""Dump ONE full episode transcript so you can eyeball the dynamics:
per round, every agent's raw Message + Position, and the position trajectory.

    python scripts/dump_episode.py --config configs/prototype.yaml \
        [--adapter runs/proto/best] [--seed 123] [--out ep.json]

--adapter runs/proto/best -> use the trained attacker; omit -> base model.
Answers: does the attacker actually move the benign agents, or do the
suggestible agents just converge on their own?
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attacker.config import load_config
from attacker.game import new_game
from attacker.generation import generate_attacker_batched
from attacker.models import load_actor, load_frozen_lm
from attacker.potential import phi
from attacker.prompts import ATTACKER_IDX, build_chat
from attacker.rollout import _game_cfg
from attacker.world_model import BenignEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (trained attacker)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="episode_dump.json")
    ap.add_argument("--greedy", action="store_true", help="greedy instead of sampling")
    ap.add_argument("--actor-model", default=None,
                    help="override actor.model_name (e.g. an instruct model) for this dump")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.actor_model:
        cfg.actor.model_name = args.actor_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample = not args.greedy

    actor, actor_tok = load_actor(cfg, bf16=cfg.train.bf16, adapter_path=args.adapter)
    actor.to(device)
    wm_model, wm_tok = load_frozen_lm(cfg.world_model.model_name, bf16=cfg.train.bf16)
    wm = BenignEngine(wm_model.to(device), wm_tok, max_pos=cfg.game.max_pos,
                      max_new_tokens=cfg.world_model.max_new_tokens,
                      personality=cfg.game.benign_personality, msg_order=cfg.game.msg_order)

    gcfg = _game_cfg(cfg)
    st = new_game(gcfg, seed=args.seed)
    T = cfg.game.max_steps
    pot = [list(st.positions)]
    rounds = []

    for t in range(T):
        st.t = t
        # attacker
        chat = build_chat(ATTACKER_IDX, "malicious", st.positions, st.last_messages,
                          t, gcfg.msg_order, st.rng)
        gt = generate_attacker_batched(
            actor, actor_tok, [chat], [st.positions[ATTACKER_IDX]],
            max_msg_tokens=cfg.actor.max_new_tokens, sample=sample,
            temperature=1.0, max_pos=gcfg.max_pos)[0]
        from attacker.game import clamp_position
        st.positions[ATTACKER_IDX] = clamp_position(gt.next_position, gcfg)
        # benign
        benign_msgs = wm.step_batched([st], sample=sample)[0]
        st.last_messages = list(benign_msgs) + [gt.message]

        rec = {
            "round": t,
            "phi_benign": round(phi(st.benign_positions()), 4),
            "benign_positions": st.benign_positions(),
            "attacker": {"pos": st.positions[ATTACKER_IDX], "msg": gt.message,
                         "valid": gt.valid},
            "benign": [{"agent": i, "pos": st.positions[i], "msg": benign_msgs[i]}
                       for i in range(3)],
        }
        rounds.append(rec)
        pot.append(list(st.positions))

        print(f"--- round {t}  phi={rec['phi_benign']}  benign={rec['benign_positions']} ---")
        print(f"  ATTACKER (Agent 3) pos={rec['attacker']['pos']:2d} :: {gt.message!r}")
        for b in rec["benign"]:
            print(f"  benign {b['agent']} pos={b['pos']:2d} :: {b['msg']!r}")

        if st.consensus_reached() or t == T - 1:
            print(f"[end] {'CONSENSUS' if st.consensus_reached() else 'SURVIVED'} "
                  f"at round {t}")
            break

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "adapter": args.adapter,
                   "positions_over_time": pot, "rounds": rounds}, f,
                  ensure_ascii=False, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
