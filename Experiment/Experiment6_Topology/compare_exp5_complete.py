#!/usr/bin/env python
"""Read-only, episode-paired comparison of Exp5 B5 and Exp6 complete JSONs."""
import argparse
from collections import Counter
import json
from pathlib import Path


def index_records(data, experiment, mode):
    records = [r for r in data["episodes"]
               if (r.get("condition") == f"B5_{mode}" if experiment == 5 else
                   r.get("topology") == "complete" and r.get("mode") == mode)]
    indexed = {r["episode"]: r for r in records}
    if not records or len(indexed) != len(records):
        raise ValueError(f"Exp{experiment}/{mode}: missing or duplicate episodes")
    return indexed


def compare(old, new):
    for key in ("seed", "max_steps", "horizons", "round_timing"):
        if old.get(key) != new.get(key):
            raise ValueError(f"incompatible {key}: {old.get(key)!r} vs {new.get(key)!r}")
    print("Historical Exp5 JSON may omit model/backend/batch metadata; verify its launch log separately.")
    all_equal = True
    for mode in new["modes"]:
        a, b = index_records(old, 5, mode), index_records(new, 6, mode)
        if set(a) != set(b):
            raise ValueError(f"{mode}: episode ID sets differ ({len(a)} vs {len(b)})")
        print(f"\n{mode}: {len(a)} paired episodes")
        for label, records in (("Exp5 B5", a), ("Exp6 complete", b)):
            n = len(records)
            c = sum(int(r["consensus"]) for r in records.values())
            length = sum(r["ep_len"] for r in records.values()) / n
            print(f"  {label:14s} CR={c/n:.4f} ({c}/{n}) mean_ep_len={length:.4f}")
        mismatches, compared = Counter(), Counter()
        first = None
        for ep in sorted(a):
            for key in ("initial_positions", "benign_personalities", "consensus", "ep_len"):
                compared[key] += 1
                if a[ep][key] != b[ep][key]:
                    mismatches[key] += 1
                    first = first or (ep, key)
            if a[ep].get("rounds") is not None and b[ep].get("rounds") is not None:
                # Compare every historical trace key. New audit fields may be absent
                # in older dumps, but missing old fields in the new dump are errors.
                compared["rounds"] += 1
                ar, br = a[ep]["rounds"], b[ep]["rounds"]
                same = len(ar) == len(br) and all(
                    all(key in y and y[key] == value for key, value in x.items())
                    for x, y in zip(ar, br))
                if not same:
                    mismatches["rounds"] += 1
                    first = first or (ep, "rounds")
        for key, count in compared.items():
            print(f"  {key:22s} mismatched episodes={mismatches[key]}/{count}")
        if not compared["rounds"]:
            print("  WARNING: no shared full traces; trajectory equivalence cannot be checked.")
        if first:
            print(f"  First mismatch: episode={first[0]}, field={first[1]}")
            all_equal = False
    print("\n" + ("MATCH on all available compared fields." if all_equal else
                    "DIFFERENCES FOUND: inspect provenance and trajectories before other topologies."))
    return all_equal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp5", required=True)
    p.add_argument("--exp6", required=True)
    args = p.parse_args()
    old, new = [json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
                for path in (args.exp5, args.exp6)]
    raise SystemExit(0 if compare(old, new) else 1)


if __name__ == "__main__":
    main()
