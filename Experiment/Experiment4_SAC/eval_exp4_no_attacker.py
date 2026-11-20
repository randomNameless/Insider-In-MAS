#!/usr/bin/env python
"""Four-benign SAC control: SAC on agents 0-2, neutral agent 3, score first 3."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(HERE.parent / "Experiment3_CPWBFT")]
sys.path.insert(0, str(HERE.parent / "Experiment1_Eval"))
from eval_attacks_consensus import QwenWorldModelVLLM  # noqa: E402
from eval_exp3_cpwbft import load_causal_model  # noqa: E402
from attacker.action import parse_pos_msg  # noqa: E402
from attacker.generation import generate_benign_batched  # noqa: E402
from attacker.prompts import build_chat  # noqa: E402
from sac_defense import SACBenignEngine, SACBenignEngineVLLM  # noqa: E402

PERSONALITIES = ["stubborn", "suggestible", "neutral", "neutral"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model_dir", required=True)
    p.add_argument("--backend", default="vllm", choices=("vllm", "hf"))
    p.add_argument("--gpu_mem_util", type=float, default=0.75)
    p.add_argument("--max_model_len", type=int, default=2048)
    p.add_argument("--n_exp", type=int, default=200); p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--batch_size", type=int, default=8); p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--max_pos", type=int, default=20); p.add_argument("--dump_json", required=True)
    p.add_argument("--log_every", type=int, default=50)
    a = p.parse_args()
    if a.backend == "vllm":
        wm = QwenWorldModelVLLM(
            model_dir=a.model_dir, max_pos=a.max_pos, max_new_tokens=200,
            gpu_memory_utilization=a.gpu_mem_util,
            max_model_len=a.max_model_len,
            chat_template_kwargs={"enable_thinking": False},
            temperature=0.7, seed=a.seed)
        sac = SACBenignEngineVLLM(
            wm, max_pos=a.max_pos, max_new_tokens=200,
            scorer_max_new_tokens=8, f=1, temperature=0.7,
            seed=a.seed + 200000)
        model = tok = None
    else:
        model, tok = load_causal_model(a.model_dir, "cuda")
        sac = SACBenignEngine(model, tok, max_pos=a.max_pos, f=1)
    records = []
    for start in range(0, a.n_exp, a.batch_size):
        states = []
        for ep in range(start, min(start + a.batch_size, a.n_exp)):
            rng = random.Random(a.seed + ep + 10000)
            states.append(SimpleNamespace(positions=[rng.randint(0, a.max_pos) for _ in range(4)],
                last_messages=[None] * 4, personalities=list(PERSONALITIES), t=0,
                rng=rng, confidences=[1.0] * 4))
        live = [True] * len(states); lengths = [0] * len(states); outcomes = [False] * len(states)
        histories = [[] for _ in states]
        while any(live):
            ids = [i for i, flag in enumerate(live) if flag]; active = [states[i] for i in ids]
            # Snapshot ensures all four updates are synchronous.
            snapshots = [(list(s.positions), list(s.last_messages)) for s in active]
            msgs012 = sac.step_batched(active, sample=True)
            chats = [build_chat(3, "neutral", pos, msgs, s.t, "index", s.rng)
                     for s, (pos, msgs) in zip(active, snapshots)]
            if a.backend == "vllm":
                texts3 = sac._generate_chats(
                    chats, max_tokens=200, temperature=0.7)
                actions3 = [parse_pos_msg(text, pos[3], a.max_pos)
                            for text, (pos, _) in zip(texts3, snapshots)]
                conf3 = [1.0] * len(actions3)
            else:
                actions3, conf3 = generate_benign_batched(
                    model, tok, chats, [pos[3] for pos, _ in snapshots],
                    max_pos=a.max_pos, max_new_tokens=200,
                    sample=True, temperature=0.7)
            for local, global_i in enumerate(ids):
                s = states[global_i]; s.positions[3] = int(actions3[local].next_position)
                s.confidences[3] = float(conf3[local])
                s.last_messages = list(msgs012[local]) + [actions3[local].message]
                s.t += 1; lengths[global_i] += 1
                outcomes[global_i] = max(s.positions[:3]) == min(s.positions[:3])
                live[global_i] = not outcomes[global_i] and lengths[global_i] < a.max_steps
                histories[global_i].append({"t": s.t - 1, "positions": list(s.positions),
                    "receiver_scores": s.sac_scores,
                    "retained_neighbors": s.sac_retained_neighbors})
        for i, s in enumerate(states):
            records.append({"episode": start + i, "consensus": outcomes[i],
                "ep_len": lengths[i], "rounds": histories[i]})
        completed = min(start + len(states), a.n_exp)
        if a.log_every > 0 and (completed == a.n_exp or completed % a.log_every == 0):
            print(f"[no-attacker SAC] completed {completed}/{a.n_exp}", flush=True)
    count = sum(int(x["consensus"]) for x in records)
    mean_len = sum(x["ep_len"] for x in records) / len(records)
    print(f"condition=sac_four_benign episodes={len(records)} consensus={count} "
          f"rate={count/len(records):.3f} mean_ep_len={mean_len:.2f}")
    out = Path(a.dump_json).expanduser(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"experiment":"4-no-attacker", "defense":"SAC_F1_on_0_1_2",
        "round_timing":"synchronous_snapshot_atomic_commit",
        "consensus_scope":[0,1,2], "seed":a.seed, "model_dir":a.model_dir,
        "wm_backend":a.backend,
        "summary":{"episodes":len(records),"consensus":count,"rate":count/len(records),
        "mean_ep_len":mean_len}, "episodes":records}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
