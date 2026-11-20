#!/usr/bin/env python
"""Per-seed results (2026-10-22): for each condition of an eval_debate run, the final seat-level accuracy and its
paired difference vs `none` separately for every evaluation seed (same questions, same seats, same honest sampling
seeds), plus mean / sample std over seeds. Also majority-on-target per seed.
  python debate_wm/per_seed_table.py RUN_DIR [RUN_DIR ...] --out per_seed.json
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debate_ppo import eval_debate as E  # noqa: E402


def per_seed(run: Path) -> dict:
    st = E.load_state(run)
    m = defaultdict(dict)                              # (cond, seed) -> task -> metrics
    for d in st["debates"]:
        if d["final"] is None:
            continue
        m[(d["cond"], d["seed"])][d["task_id"]] = E.debate_metrics(d)
    seeds = sorted({s for _, s in m})
    out = {}
    for c in st["config"]["conds"]:
        rows = {}
        for s in seeds:
            cur = m[(c, s)]
            acc = sum(x["final_acc"] for x in cur.values()) / len(cur)
            row = {"n": len(cur), "final_acc": acc}
            if c != "none":
                base = m[("none", s)]
                qs = sorted(set(cur) & set(base))
                row["d_vs_none"] = sum(cur[q]["final_acc"] - base[q]["final_acc"] for q in qs) / len(qs)
                tg = [x["majority_on_target"] for x in cur.values() if "majority_on_target" in x]
                row["majority_on_target"] = sum(tg) / len(tg) if tg else None
            rows[s] = row
        summ = {}
        for k in ["final_acc", "d_vs_none", "majority_on_target"]:
            v = [r[k] for r in rows.values() if r.get(k) is not None]
            if v:
                summ[k] = {"mean": statistics.mean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0,
                           "min": min(v), "max": max(v)}
        out[c] = {"per_seed": rows, "summary": summ}
    return {"run": str(run), "seeds": seeds, "conditions": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = [per_seed(Path(r)) for r in a.runs]
    for r in res:
        print(f"== {r['run']} seeds={r['seeds']}")
        for c, v in r["conditions"].items():
            ps = " ".join(f"s{s}:{x['final_acc']:.3f}" + (f"({x['d_vs_none']:+.3f})" if "d_vs_none" in x else "")
                          for s, x in v["per_seed"].items())
            sm = v["summary"].get("d_vs_none")
            print(f"  {c:24s} {ps}" + (f"  | d mean {sm['mean']:+.3f} sd {sm['std']:.3f}" if sm else ""))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
