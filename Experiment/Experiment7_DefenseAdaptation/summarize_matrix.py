#!/usr/bin/env python3
"""Verify Exp7 manifests/outcomes; print CR and censored-length matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from run_matrix import digest, read, require, sha, validate_result

ROWS = ["exp1_nodefense", "exp2_wmsr", "exp3_cpwbft", "exp4_curriculum"]
COLS = ["no_defense", "wmsr", "cpwbft", "sac"]


def measure(xs, digits=3):
    sd = f"{stdev(xs):.{digits}f}" if len(xs) > 1 else "N/A"
    return f"{mean(xs):.{digits}f} ± {sd}"


def summarize(root, seeds):
    cells, protocols, identities = {}, set(), {}
    for path in sorted(root.glob("*/*/*/eval_seed*.manifest.json")):
        m = read(path)
        require(m.get("status") == "complete", f"Incomplete job: {path}")
        s = m["spec"]
        if s["seed"] not in seeds:
            continue
        require(m["job_id"] == digest(s), f"Manifest spec changed: {path}")
        result = path.with_name(path.name.replace(".manifest.json", ".json"))
        require(result.is_file() and sha(result) == m["result_sha256"], f"Missing/changed result: {result}")
        stats = validate_result(result, s)
        victim, adapter, defense, seed = s["victim"], s["adapter"], s["defense"], s["seed"]
        require(adapter in ROWS and defense in COLS, f"Unknown cell: {path}")
        require(result.relative_to(root).parts == (victim, adapter, defense, f"eval_seed{seed}.json"),
                f"Path/manifest cell mismatch: {path}")
        key = (victim, adapter, defense, seed)
        require(key not in cells, f"Duplicate cell: {key}")
        cells[key] = stats
        # Seeds/adapter/defense vary; protocol, base, runtime and source must not.
        protocols.add(digest({k: s[k] for k in ("protocol", "settings", "base", "source_sha256",
                                               "python_version", "environment_freeze")}))
        for name, identity in (("adapter:" + adapter, s["adapter_info"]),
                               ("victim:" + victim, s["victim_info"])):
            require(name not in identities or identities[name] == identity, f"Mixed identity: {name}")
            identities[name] = identity
    require(cells, "No completed manifested Exp7 cells found")
    require(len(protocols) == 1, "Mixed protocol/code/environment: use separate result directories")
    lines = ["# Experiment 7 — Defense-Specific Adaptation", "",
             f"Seeds: {', '.join(map(str, seeds))}. Mean ± sample SD across seed means.",
             "CR lower = stronger attacker. Length includes failures at the run horizon.",
             "Missing cells are NOT zeros. Diagonal is not assumed to be best.", ""]
    missing = []
    for victim in sorted({key[0] for key in cells}):
        lines += [f"## Victim: {victim}", ""]
        for title, metric, digits in (("Consensus rate", "rate", 3), ("Mean episode length", "mean_ep_len", 2)):
            lines += [f"### {title}", "", "| Trained adapter | " + " | ".join(COLS) + " |",
                      "|---|" + "---|" * 4]
            for adapter in ROWS:
                vals = []
                for defense in COLS:
                    keys = [(victim, adapter, defense, seed) for seed in seeds]
                    if any(k not in cells for k in keys):
                        vals.append("INCOMPLETE")
                        missing.extend(k for k in keys if k not in cells)
                    else:
                        vals.append(measure([cells[k][metric] for k in keys], digits))
                lines.append("| " + " | ".join([adapter] + vals) + " |")
            lines.append("")
        lines += ["### Per-cell seed results and pooled CR", "",
                  "| Adapter | Defense | CR per seed | Pooled CR |",
                  "|---|---|---|---|"]
        for adapter in ROWS:
            for defense in COLS:
                keys = [(victim, adapter, defense, seed) for seed in seeds]
                if any(k not in cells for k in keys):
                    continue
                rr = [cells[k] for k in keys]
                count, n = sum(x["consensus"] for x in rr), sum(x["episodes"] for x in rr)
                rates = ", ".join(f"{x['rate']:.3f}" for x in rr)
                lines.append(f"| {adapter} | {defense} | {rates} | {count/n:.3f} ({count}/{n}) |")
        lines += ["", "### Same-column gain over the Exp1 adapter", "",
                  "Gain = CR(Exp1 adapter, defense) − CR(row adapter, same defense). Positive is better than Exp1; NOT a no-attack effect.",
                  "", "| Adapter | " + " | ".join(COLS) + " |", "|---|" + "---|" * 4]
        for adapter in ROWS[1:]:
            vals = []
            for defense in COLS:
                pairs = [((victim, "exp1_nodefense", defense, seed), (victim, adapter, defense, seed)) for seed in seeds]
                vals.append(measure([cells[n]["rate"] - cells[p]["rate"] for n, p in pairs])
                            if all(n in cells and p in cells for n, p in pairs) else "INCOMPLETE")
            lines.append("| " + " | ".join([adapter] + vals) + " |")
        lines.append("")
    lines += ["## Adapter identities", ""]
    for key, info in sorted(identities.items()):
        if key.startswith("adapter:"):
            lines += [f"- {key}: `{info['path']}`; SHA256 `{info['weight_sha256']}`"]
    lines += ["", f"Missing cell/seed combinations: {len(set(missing))}."]
    return "\n".join(lines) + "\n", sorted(set(missing))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", type=Path, default=Path(__file__).parent / "results/matrix_v1")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(9000, 9005)))
    p.add_argument("--output", type=Path, help="Optional new Markdown report; existing file is not overwritten")
    a = p.parse_args()
    require(a.seeds and len(set(a.seeds)) == len(a.seeds), "Duplicate/empty seeds")
    report, missing = summarize(a.result_dir.expanduser().resolve(), a.seeds)
    print(report)
    if a.output:
        with a.output.expanduser().open("x", encoding="utf-8") as f:
            f.write(report)
    if missing:
        print("Incomplete matrix; finish the missing jobs before reporting the full table.")


if __name__ == "__main__":
    main()
