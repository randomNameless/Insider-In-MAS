#!/usr/bin/env python
"""Summarize five-seed CR and episode length for Exp1 PPO under defenses."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="~/consensus-attacker/Experiment/results/exp1_ppo_transfer_to_defenses")
    p.add_argument("--seeds", type=int, nargs="+", default=[9000, 9001, 9002, 9003, 9004])
    args = p.parse_args()
    root = Path(args.root).expanduser()
    defenses = [("exp2_wmsr", "W-MSR"), ("exp3_cpwbft", "CP-WBFT"),
                ("exp4_sac", "SAC")]
    models = [("wm", "qwen35-wm"), ("qwen35_9b", "Qwen3.5-9B")]

    print(f"{'defense':10s} {'benign model':13s} {'seed CRs':34s} "
          f"{'CR mean +/- sd':18s} {'pooled CR':17s} {'mean ep len +/- sd [range]'}")
    print("-" * 125)
    for defense_dir, defense_name in defenses:
        for model_dir, model_name in models:
            rows, missing = [], []
            for seed in args.seeds:
                path = root / defense_dir / model_dir / f"eval_seed{seed}.json"
                if not path.is_file():
                    missing.append(str(path))
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                row = payload["summary"]["model_rl"]
                rows.append((int(row["episodes"]), int(row["consensus"]),
                             float(row["rate"]), float(row["mean_ep_len"])))
            if missing:
                print(f"{defense_name:10s} {model_name:13s} MISSING {len(missing)}/{len(args.seeds)}")
                for path in missing:
                    print(f"  {path}")
                continue
            rates = [row[2] for row in rows]
            lengths = [row[3] for row in rows]
            n = sum(row[0] for row in rows)
            successes = sum(row[1] for row in rows)
            rate_sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
            len_sd = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
            print(f"{defense_name:10s} {model_name:13s} "
                  f"{', '.join(f'{x:.3f}' for x in rates):34s} "
                  f"{statistics.mean(rates):.3f} +/- {rate_sd:.3f}  "
                  f"{successes/n:.3f} ({successes}/{n})  "
                  f"{statistics.mean(lengths):.2f} +/- {len_sd:.2f} "
                  f"[{min(lengths):.2f}, {max(lengths):.2f}]")


if __name__ == "__main__":
    main()
