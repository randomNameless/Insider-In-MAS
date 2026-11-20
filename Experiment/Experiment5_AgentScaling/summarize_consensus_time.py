#!/usr/bin/env python
"""Print time-to-consensus and right-censored failures from one Exp5 JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json")
    p.add_argument("--condition", default="B4_M1")
    args = p.parse_args()
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    row = data["summary"][args.condition]
    print(f"condition={args.condition}")
    print(f"episodes={row['episodes']} max_steps={row['max_steps']}")
    print(f"consensus={row['consensus']} rate={row['rate']:.3f}")
    print(f"nonconsensus_at_horizon={row['nonconsensus_at_horizon']}")
    print(f"mean_round_to_consensus={row['mean_round_to_consensus']}")
    print(f"median_round_to_consensus={row['median_round_to_consensus']}")
    print("\nconsensus round : episode count")
    for step, count in row["consensus_round_histogram"].items():
        if count:
            print(f"{int(step):3d} : {count}")


if __name__ == "__main__":
    main()
