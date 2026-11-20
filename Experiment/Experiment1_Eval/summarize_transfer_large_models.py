#!/usr/bin/env python
"""Summarise no-attack plus six-attacker transfer JSONs across seeds."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ATTACKS = ("no_attack", "random", "extreme", "prompt", "dqn",
           "model_base", "model_rl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_root", required=True)
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[9000, 9001, 9002, 9003, 9004])
    args = p.parse_args()
    root = Path(args.result_root)

    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        per_seed = {}
        for seed in args.seeds:
            no = json.loads((model_dir / "no_attack" /
                             f"eval_seed{seed}.json").read_text())
            attacked = json.loads((model_dir / "attacks" /
                                   f"eval_seed{seed}.json").read_text())
            per_seed[seed] = {"no_attack": no["summary"],
                              **attacked["summary"]}

        print(f"\nMODEL: {model_dir.name}")
        print("condition    rates                              mean +/- sd       pooled rate       mean ep len")
        print("-" * 108)
        for attack in ATTACKS:
            rows = [per_seed[seed][attack] for seed in args.seeds]
            rates = [float(row["rate"]) for row in rows]
            lens = [float(row["mean_ep_len"]) for row in rows]
            count = sum(int(row["consensus"]) for row in rows)
            total = sum(int(row["episodes"]) for row in rows)
            sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
            print(f"{attack:12s} "
                  f"{', '.join(f'{x:.3f}' for x in rates):34s} "
                  f"{statistics.mean(rates):.3f} +/- {sd:.3f}   "
                  f"{count / total:.3f} ({count}/{total})   "
                  f"{statistics.mean(lens):.2f}")


if __name__ == "__main__":
    main()

