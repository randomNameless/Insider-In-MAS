#!/usr/bin/env python
"""Compare WM message text with the 9B's real message (eval_wm.py openloop files with "messages").

cos    = mean cosine(bge-m3 embedding of a WM sample, embedding of the real 9B message), 8 samples/turn;
         the 9B's own resamples give the ceiling.
stay%  = share of samples whose text signals staying put; follow% = signals following others.
words  = mean message length. Grouped by persona and round bucket (1, 2+).

    python -m consensus_wm.analyze_msgs --ref ref9b_shuffle.jsonl A=A_shuffle.jsonl C=C_shuffle.jsonl --examples 6
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict

import numpy as np

STAY = re.compile(r"\b(stay|staying|not moving|won't move|remain|right here|hold|keep my)\b", re.I)
FOLLOW = re.compile(r"\b(follow|join|happy to|glad to|go wherever|move toward|heading to|come to)\b", re.I)


def rows(path):
    return {(r["gid"], r["t"], r["agent"]): r for r in map(json.loads, open(path))}


def main(argv):
    ref = rows(argv[argv.index("--ref") + 1])
    arms = [x.split("=", 1) for x in argv if "=" in x]
    n_ex = int(argv[argv.index("--examples") + 1]) if "--examples" in argv else 0
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("/path/to/models/bge-m3", device="cuda")

    def embed(texts):
        return enc.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)

    keys = sorted(ref)
    true_msgs = [ref[k]["true_message"] for k in keys]
    E_true = embed(true_msgs)
    print("| arm | group | n | cos | stay% | follow% | words |")
    print("|---|---|---|---|---|---|---|")
    for name, path in [("9B-self", None)] + arms:
        W = ref if path is None else rows(path)
        flat = [m for k in keys for m in W[k]["messages"]]
        E = embed(flat).reshape(len(keys), -1, E_true.shape[1])
        cos = (E * E_true[:, None, :]).sum(-1).mean(1)
        groups = defaultdict(list)
        for i, k in enumerate(keys):
            r = ref[k]
            ms = W[k]["messages"]
            rec = (cos[i], np.mean([bool(STAY.search(m)) for m in ms]), np.mean([bool(FOLLOW.search(m)) for m in ms]),
                   np.mean([len(m.split()) for m in ms]))
            b = "r1" if r["t"] == 0 else "r2+"
            for g in ("all", r["persona"], b, f"{r['persona']} {b}"):
                groups[g].append(rec)
        for g in ["all", "stubborn", "suggestible", "neutral", "r1", "r2+",
                  "stubborn r1", "stubborn r2+", "suggestible r1", "suggestible r2+", "neutral r1", "neutral r2+"]:
            a = np.array(groups[g])
            print(f"| {name} | {g} | {len(a)} | {a[:, 0].mean():.3f} | {100 * a[:, 1].mean():.0f} | "
                  f"{100 * a[:, 2].mean():.0f} | {a[:, 3].mean():.1f} |")
    if n_ex:
        rng = random.Random(0)
        W = {name: rows(path) for name, path in arms}
        cand = [k for k in keys if ref[k]["t"] >= 1]
        print("\n## examples (round >= 2)")
        for persona in ("stubborn", "suggestible", "neutral"):
            for k in rng.sample([c for c in cand if ref[c]["persona"] == persona], n_ex // 3):
                r = ref[k]
                print(f"\n**{persona}, game {k[0]}, round {k[1] + 1}, Agent {k[2]}, at {r['cur']} -> {r['true_next']}**")
                print(f"- 9B real: {r['true_message']}")
                for name in W:
                    w = W[name][k]
                    print(f"- {name}: {w['messages'][0]}  (pos {w['samples'][0]})")


if __name__ == "__main__":
    main(sys.argv[1:])
