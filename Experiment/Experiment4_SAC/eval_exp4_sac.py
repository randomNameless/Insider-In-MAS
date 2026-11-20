#!/usr/bin/env python
"""Synchronous SAC evaluation for all six Experiment-1 attackers."""
from __future__ import annotations

import argparse, json, random, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP3 = HERE.parent / "Experiment3_CPWBFT"
sys.path[:0] = [str(ROOT), str(EXP3)]

from eval_exp3_cpwbft import ConfidenceCPEnv, ModelHolder, load_causal_model  # noqa: E402
from eval_attacks_consensus import (  # noqa: E402
    HFQwen3, MIXED_PERSONALITIES, QwenWorldModelVLLM,
    load_dqn, make_attacker_fn,
)
from sac_defense import SACBenignEngine, SACBenignEngineVLLM, self_check  # noqa: E402

ATTACKS = ("random", "extreme", "prompt", "dqn", "model_base", "model_rl")


def _json_default(value):
    """Convert NumPy/Torch scalar containers produced by the environments."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    self_check()
    p = argparse.ArgumentParser()
    p.add_argument("--wm_model_dir", required=True)
    p.add_argument("--wm_backend", default="vllm", choices=("vllm", "hf"))
    p.add_argument("--wm_gpu_mem_util", type=float, default=0.65)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--llm_model_dir", required=True)
    p.add_argument("--rl_adapter", required=True)
    p.add_argument("--dqn_ckpt", default=None)
    p.add_argument("--dqn_device", default="cpu")
    p.add_argument("--attacks", default=",".join(ATTACKS))
    p.add_argument("--n_exp", type=int, default=200)
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--max_pos", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--actor_max_message_tokens", type=int, default=48)
    p.add_argument("--wm_max_new_tokens", type=int, default=200)
    p.add_argument("--scorer_max_new_tokens", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dump_json", required=True)
    p.add_argument("--dump_n", type=int, default=200)
    p.add_argument("--log_every", type=int, default=50)
    a = p.parse_args()
    attacks = [x.strip() for x in a.attacks.split(",") if x.strip()]
    if set(attacks) - set(ATTACKS):
        p.error(f"allowed attacks: {ATTACKS}")
    if "dqn" in attacks and not a.dqn_ckpt:
        p.error("the dqn attacker requires --dqn_ckpt")

    if a.wm_backend == "vllm":
        wm_backend = QwenWorldModelVLLM(
            model_dir=a.wm_model_dir, max_pos=a.max_pos,
            max_new_tokens=a.wm_max_new_tokens,
            gpu_memory_utilization=a.wm_gpu_mem_util,
            max_model_len=a.max_model_len, enforce_eager=a.enforce_eager,
            chat_template_kwargs={"enable_thinking": False},
            temperature=0.7, seed=a.seed)
        sac = SACBenignEngineVLLM(
            wm_backend, max_pos=a.max_pos,
            max_new_tokens=a.wm_max_new_tokens,
            scorer_max_new_tokens=a.scorer_max_new_tokens,
            f=1, temperature=0.7, seed=a.seed + 200000)
        holder = ModelHolder(None, wm_backend.tokenizer, a.max_pos)
    else:
        wm, wm_tok = load_causal_model(a.wm_model_dir, a.device)
        sac = SACBenignEngine(
            wm, wm_tok, max_pos=a.max_pos,
            max_new_tokens=a.wm_max_new_tokens,
            scorer_max_new_tokens=a.scorer_max_new_tokens, f=1)
        holder = ModelHolder(wm, wm_tok, a.max_pos)
    # Reuse the exact Experiment-1 attacker implementations. SAC does not use
    # sender-reported confidence, so Random/Extreme/DQN are valid baselines.
    llm = HFQwen3(
        a.llm_model_dir, adapter_dir=a.rl_adapter, device=a.device,
        max_new_tokens=a.actor_max_message_tokens, temperature=0.7, seed=a.seed)
    dqn_device = torch.device(a.dqn_device)
    q_net = load_dqn(a.dqn_ckpt, dqn_device) if "dqn" in attacks else None
    salts = {"random": 11, "extreme": 23, "prompt": 37,
             "dqn": 53, "model_base": 67, "model_rl": 71}
    attacker_fns = {
        kind: make_attacker_fn(
            kind, llm, q_net, dqn_device,
            random.Random(a.seed + salts[kind]), a.max_pos)
        for kind in attacks
    }

    def env():
        return ConfidenceCPEnv(world_model=holder, max_pos=a.max_pos,
            max_steps=a.max_steps, consensus_tol=0, consensus_penalty=-10.0,
            reward_mode="spread", attacker_msg_mode="template")

    initial = []
    for ep in range(a.n_exp):
        rng = random.Random(a.seed + ep + a.max_steps * 1000)
        initial.append([rng.randint(0, a.max_pos) for _ in range(4)])
    summaries, records = {}, []
    for kind in attacks:
        random.seed(a.seed + 100000); np.random.seed((a.seed + 100000) % 2**32)
        torch.manual_seed(a.seed + 100000); torch.cuda.manual_seed_all(a.seed + 100000)
        if hasattr(sac, "set_seed"):
            sac.set_seed(a.seed + 200000)
        outcomes = []
        for start in range(0, a.n_exp, a.eval_batch_size):
            batch = [env() for _ in initial[start:start + a.eval_batch_size]]
            for e, pos in zip(batch, initial[start:start + len(batch)]):
                e.reset(options={"positions": pos, "personalities": list(MIXED_PERSONALITIES)})
            live = [True] * len(batch)
            rounds = [[] for _ in batch]
            lengths = [0] * len(batch)
            consensus = [False] * len(batch)
            while any(live):
                ids = [i for i, flag in enumerate(live) if flag]
                envs = [batch[i] for i in ids]
                outputs = attacker_fns[kind].batch(envs)
                gens = [SimpleNamespace(next_position=int(pos), message=message)
                        for pos, message in outputs]
                states = []
                for local, (e, gen) in enumerate(zip(envs, gens)):
                    states.append(SimpleNamespace(
                        positions=list(map(int, e.pos)),
                        sac_snapshot_positions=list(map(int, e.pos)),
                        last_messages=list(e.last_messages),
                        personalities=list(e.personalities), t=int(e.t),
                        rng=random.Random(a.seed + start + local),
                        confidences=[1.0] * 4))
                benign_msgs = sac.step_batched(states, sample=True)
                for local, global_i in enumerate(ids):
                    e, state, gen = envs[local], states[local], gens[local]
                    e.pos[:3] = state.positions[:3]
                    e.pos[3] = int(gen.next_position)
                    e.last_messages = list(benign_msgs[local]) + [gen.message]
                    e.t += 1; lengths[global_i] += 1
                    consensus[global_i] = bool(max(e.pos[:3]) == min(e.pos[:3]))
                    live[global_i] = not consensus[global_i] and lengths[global_i] < a.max_steps
                    if start + global_i < a.dump_n:
                        rounds[global_i].append({
                            "t": lengths[global_i] - 1,
                            "attacker_position": int(gen.next_position),
                            "attacker_message": gen.message,
                            "round_snapshot_positions": state.sac_snapshot_positions,
                            "benign_positions": list(map(int, e.pos[:3])),
                            "receiver_scores": state.sac_scores,
                            "retained_neighbors": state.sac_retained_neighbors,
                            "consensus_after": bool(consensus[global_i])})
            for local in range(len(batch)):
                outcomes.append((consensus[local], lengths[local]))
                ep = start + local
                if ep < a.dump_n:
                    records.append({"attacker": kind, "episode": ep,
                        "initial_positions": initial[ep],
                        "consensus": bool(consensus[local]),
                        "ep_len": lengths[local], "rounds": rounds[local]})
            completed = min(start + len(batch), a.n_exp)
            if a.log_every > 0 and (completed == a.n_exp or completed % a.log_every == 0):
                print(f"[{kind}] completed {completed}/{a.n_exp} episodes", flush=True)
        count = sum(int(x[0]) for x in outcomes)
        summaries[kind] = {"episodes": len(outcomes), "consensus": count,
            "rate": count / len(outcomes),
            "mean_ep_len": sum(x[1] for x in outcomes) / len(outcomes)}

    print("\n" + "#" * 74 + "\n# EXPERIMENT 4: SAC evaluation\n" + "#" * 74)
    for kind in attacks:
        row = summaries[kind]
        print(f"attacker={kind:10s} episodes={row['episodes']:4d} "
              f"consensus={row['consensus']:4d} rate={row['rate']:.3f} "
              f"mean_ep_len={row['mean_ep_len']:.2f}")
    out = Path(a.dump_json).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiment": 4, "defense": "SAC_F1_complete4",
        "round_timing": "synchronous_snapshot_atomic_commit",
        "seed": a.seed, "wm_model_dir": a.wm_model_dir,
        "wm_backend": a.wm_backend,
        "summary": summaries, "episodes": records}, indent=2,
        default=_json_default), encoding="utf-8")
    print(f"[dump] wrote {len(records)} trajectories -> {out}")


if __name__ == "__main__":
    main()
