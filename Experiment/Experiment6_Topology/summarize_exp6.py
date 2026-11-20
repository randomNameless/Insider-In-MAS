#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", required=True)
    p.add_argument("--seeds", nargs="*", type=int,
                   default=[9000, 9001, 9002, 9003])
    args = p.parse_args()
    root = Path(args.result_dir)
    payloads = [json.loads((root / f"eval_seed{s}.json").read_text())
                for s in args.seeds]
    if payloads[0].get("design_version") in {"paired_topology_v4", "paired_topology_6nodes_v5", "paired_topology_6nodes_exp5_aligned_v6"}:
        summarize_paired(payloads)
        return
    names = list(payloads[0]["summary"])
    print("topology                rates                         mean +/- sd       pooled rate      mean ep len")
    print("-" * 116)
    for name in names:
        rows = [data["summary"][name] for data in payloads]
        rates = [float(row["rate"]) for row in rows]
        total_c = sum(int(row["consensus"]) for row in rows)
        total_n = sum(int(row["episodes"]) for row in rows)
        lengths = [float(row["mean_ep_len"]) for row in rows]
        sd = statistics.stdev(rates) if len(rates) > 1 else 0.0
        print(f"{name:23s} "
              f"{', '.join(f'{x:.3f}' for x in rates):29s} "
              f"{statistics.mean(rates):.3f} +/- {sd:.3f}   "
              f"{total_c / total_n:.3f} ({total_c}/{total_n})   "
              f"{statistics.mean(lengths):.2f}")


def summarize_paired(payloads):
    first = payloads[0]
    for data in payloads:
        if (data.get("design_version") != first["design_version"]
                or data["graphs"] != first["graphs"]
                or data["horizons"] != first["horizons"]
                or set(data["summary"]) != set(first["summary"])):
            raise ValueError("incompatible designs/graphs/horizons/conditions across seeds")
        for key in ("wm_model_dir", "llm_model_dir", "rl_adapter", "eval_batch_size",
                    "wm_temperature", "llm_temperature", "initial_state_seed_horizon"):
            if data["config"].get(key) != first["config"].get(key):
                raise ValueError(f"incompatible config: {key}")

    def fmt(values, digits=3):
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{statistics.mean(values):.{digits}f} +/- {sd:.{digits}f}"

    for name in first["graphs"]:
        print(f"\nTOPOLOGY: {name}")
        print(" N  mode        CR mean +/- SD    pooled CR         length mean +/- SD")
        for h in first["horizons"]:
            grouped = {}
            for mode in first["modes"]:
                rows = [d["summary"][f"{name}/{mode}"] for d in payloads]
                metrics = [r["by_horizon"][str(h)] for r in rows]
                rates = [r["rate"] for r in metrics]
                lengths = [r["mean_ep_len"] for r in metrics]
                n = sum(r["episodes"] for r in rows)
                c = sum(r["consensus"] for r in metrics)
                grouped[mode] = (rates, lengths)
                print(f"{h:2d}  {mode:10s}  {fmt(rates):18s} {c/n:.3f} ({c}/{n})     {fmt(lengths, 2)}")
            if {"no_attack", "ppo"} <= grouped.keys():
                effects = [n-p for n,p in zip(grouped["no_attack"][0], grouped["ppo"][0])]
                delays = [p-n for n,p in zip(grouped["no_attack"][1], grouped["ppo"][1])]
                print(f"    CR effect (no-PPO): {fmt(effects)}; length delta (PPO-no): {fmt(delays, 2)}")


if __name__ == "__main__":
    main()
