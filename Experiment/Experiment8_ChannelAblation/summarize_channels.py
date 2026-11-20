#!/usr/bin/env python3
"""Validate recorded channel selection and summarize across evaluation seeds."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev

from channel_game import CHANNELS, select_action


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(data):
    require(data["protocol"] == "channel_replacement_v1", "Wrong protocol")
    require(data["round_timing"] == "synchronous_snapshot_atomic_commit", "Wrong timing")
    require(data["max_steps"] == 10 and data["consensus_scope"] == [0, 1, 2], "Wrong endpoint")
    groups = defaultdict(list)
    for ep in data["episodes"]:
        require(ep["condition"] in CHANNELS, "Unknown condition")
        require(type(ep["consensus"]) is bool, "Nonboolean consensus")
        require(type(ep["ep_len"]) is int and 1 <= ep["ep_len"] <= 10, "Invalid length")
        require(len(ep["rounds"]) == ep["ep_len"], "Incomplete trajectory")
        require(ep["consensus"] or ep["ep_len"] == 10, "Early non-consensus stop")
        previous = None
        for i, r in enumerate(ep["rounds"]):
            require(r["t"] == i and r["snapshot"]["t"] == i, "Invalid round index")
            if previous is None:
                require(r["snapshot"]["positions"] == ep["initial_positions"], "Wrong initial positions")
            else:
                require(r["snapshot"] == previous, "Broken committed-state chain")
            require(r["submitted_agent3"] == select_action(ep["condition"], r["ppo_candidate"],
                                                           r["neutral_candidate"]), "Wrong channel selection")
            require([r["message_source"], r["position_source"]] == list(CHANNELS[ep["condition"]]), "Wrong source label")
            actions = r["benign_outputs"] + [r["submitted_agent3"]]
            require(len(actions) == 4, "Invalid agent count")
            require(r["after"]["positions"] == [x["position"] for x in actions], "Wrong committed positions")
            require(r["after"]["messages"] == [x["message"] for x in actions], "Wrong committed messages")
            consensus = len(set(r["after"]["positions"][:3])) == 1
            require(r["consensus_after"] == consensus, "Wrong consensus flag")
            require(not consensus or i == ep["ep_len"] - 1, "Continued after consensus")
            require(r["after"]["t"] == i + 1, "Invalid commit round")
            require(r["inputs"]["fallback_positions"] == r["snapshot"]["positions"], "Wrong generator state")
            previous = r["after"]
        require(ep["consensus"] == ep["rounds"][-1]["consensus_after"], "Wrong episode outcome")
        groups[ep["condition"]].append(ep)
    require(set(groups) == set(data["summary"]), "Summary/record conditions differ")
    by_condition = {}
    for condition, episodes in groups.items():
        require(len(episodes) == data["n_exp"], "Missing episodes")
        require(sorted(ep["exp_idx"] for ep in episodes) == list(range(data["n_exp"])), "Duplicate/missing IDs")
        initial = {ep["exp_idx"]: ep["initial_positions"] for ep in episodes}
        if by_condition:
            require(initial == next(iter(by_condition.values())), "Conditions use different initial positions")
        by_condition[condition] = initial
        count = sum(ep["consensus"] for ep in episodes)
        expected = {"episodes": len(episodes), "consensus": count,
                    "rate": count / len(episodes), "mean_ep_len": mean(ep["ep_len"] for ep in episodes)}
        for k, value in expected.items():
            require(math.isclose(data["summary"][condition][k], value, abs_tol=1e-9), f"Wrong summary {condition}/{k}")
    return data["summary"]


def formatted(values):
    return f"{mean(values):.3f} +/- {stdev(values):.3f}" if len(values) > 1 else f"{mean(values):.3f} +/- N/A"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", type=Path, default=Path(__file__).parent / "results/channel_v1")
    p.add_argument("--seeds", nargs="+", type=int, default=[9000, 9001, 9002, 9003, 9004])
    p.add_argument("--victims", nargs="+", default=["wm", "qwen35_9b"])
    a = p.parse_args()
    require(len(a.seeds) == len(set(a.seeds)), "Duplicate seeds")
    for victim in a.victims:
        results, protocol = [], None
        for seed in a.seeds:
            path = a.result_dir / victim / f"eval_seed{seed}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            require(data["victim"] == victim and data["seed"] == seed, f"Wrong seed/victim: {path}")
            stats = validate(data)
            require(set(stats) == set(CHANNELS), "Need all four conditions for comparison")
            settings = {k: v for k, v in data["settings"].items() if k not in ("out", "seeds", "check")}
            signature = {"settings": settings, "provenance": data["provenance"]}
            if protocol is not None:
                require(signature == protocol, "Mixed settings/source/adapter/environment; summarize separately")
            protocol = signature
            results.append(stats)
        print(f"\nVICTIM: {victim}; seeds={a.seeds}; SD across seed means (not episode SD)")
        print("condition              CR mean +/- SD        pooled CR        length mean +/- SD     CR(no)-CR(condition)")
        print("-" * 118)
        for condition in CHANNELS:
            rates = [s[condition]["rate"] for s in results]
            lengths = [s[condition]["mean_ep_len"] for s in results]
            effects = [s["no_attack"]["rate"] - s[condition]["rate"] for s in results]
            total = sum(s[condition]["episodes"] for s in results)
            count = sum(s[condition]["consensus"] for s in results)
            print(f"{condition:22s} {formatted(rates):21s} {count/total:.3f} ({count}/{total})   "
                  f"{formatted(lengths):22s} {formatted(effects)}")
        print("Length includes unsuccessful episodes as 10 rounds; positive CR effect means less consensus.")


if __name__ == "__main__":
    main()
