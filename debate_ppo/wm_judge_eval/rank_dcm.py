#!/usr/bin/env python
"""Score the DCM Best-of-8 oracle test (rank_bon.py's 200 sd120 states x 8 candidates) with a DCM belief WM locally, so that a
re-fit WM (other stance judge, e.g. Qwen3-4B) can be put in the same ranker table as WM_post / WM_pre of rows_lambda.json.

Input state = exactly ppo_lambda_bon.load_states (honest seats' round-r reply, stated answer and TRUE recorded belief vector,
previous round if r = 2, the 4th seat's own previous message); score = PPOBeliefWM.score(state, candidate texts), i.e. the
"tb" input the DCM's own offline test used. Output rank_bon/<tag>__WM_<variant>.jsonl, one row per (state, candidate, honest
listener) {id: "<sid>|c<c>|a<seat>", probs, p_target}, which rank_bon.py --metrics averages over listeners.
--check compares the mean P(target) per candidate with rows_lambda.json's WM_<variant>_ptarget (sanity of the local path:
with the default 1.5B judge and d_model.pt the values must agree up to bf16 batching noise).

Run from dcm_analysis with the Debate venv (see run_dcm.py):
  cd /path/to/latent_agent_transfer/dcm_analysis && python \
     <repo>/debate_ppo/wm_judge_eval/rank_dcm.py --variant post --tag dcm15 --check
  ... --variant post --judge Qwen/Qwen3-4B --judge-rev <sha> --model-suffix _q3_4b --tag dcm4b
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from debate_ppo import dprotocol as D  # noqa: E402
from debate_ppo.wm_judge_eval.rank_bon import LATENT as BON, OUT, load_states  # noqa: E402
from debate_ppo.wm_judge_eval.run_dcm import LATENT, _pad, sha256_file  # noqa: E402

LETTERS = D.LETTERS


def state_in(s):
    """= ppo_lambda_bon.load_states: round r (1-based) state with the honest seats' recorded round-r replies/beliefs."""
    d, r = s["d"], s["round"]
    seat = d["seat"]
    honest = {}
    for k, i in enumerate(D.honest_seats(seat)):
        x = d["history"][r - 1][i]
        h = {"persona": d["personalities"][i], "message": x["raw"] or f"Argument: {x['argument']}\nAnswer: {x['answer']}",
             "answer": x["answer"], "probs": _pad(d["beliefs"][r - 1][k])}
        if r >= 2:
            h.update({"prev_message": d["history"][r - 2][i]["raw"], "prev_answer": d["history"][r - 2][i]["answer"],
                      "prev_probs": _pad(d["beliefs"][r - 2][k])})
        honest[i] = h
    return {"question_id": d["task_id"], "round": r, "fourth_seat": seat, "gold": d["gold"], "target": d["target"],
            "honest": honest, "fourth_prev_message": d["history"][r - 2][seat]["raw"] if r >= 2 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("post", "pre"), required=True)
    ap.add_argument("--tag", required=True, help="output file <tag>__WM_<variant>.jsonl under rank_bon/")
    ap.add_argument("--judge", default="")
    ap.add_argument("--judge-rev", default="")
    ap.add_argument("--model-suffix", default="")
    ap.add_argument("--check", action="store_true", help="compare with rows_lambda.json WM_<variant>_ptarget")
    a = ap.parse_args()

    states = load_states()
    os.chdir(LATENT)
    sys.path.insert(0, str(LATENT))
    data_dir = LATENT / ("data/mmlupro_ppo" if a.variant == "post" else "data/mmlupro_ppo_pre")
    os.environ.setdefault("DCM_DATA", str(Path("data") / data_dir.name))
    os.environ.setdefault("DCM_PREFIX", "mmluproppo_")
    os.environ.setdefault("DCM_FIT_DATASETS", "mmlupro")
    if a.judge:
        os.environ["DCM_JUDGE"] = a.judge
        if a.judge_rev:
            os.environ["DCM_JUDGE_REV"] = a.judge_rev
    import mmlupro.ppo_bon as PB
    model_path = LATENT / ("results/abcd_" + data_dir.name) / f"d_model{a.model_suffix}.pt"
    t0 = time.perf_counter()
    wm = PB.PPOBeliefWM(model_path=model_path, data_dir=data_dir)
    load_s = time.perf_counter() - t0
    print(f"{len(states)} states, judge {PB.SS.NAME} rev {PB.SS.REV}, model {model_path}, load {load_s:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.tag}__WM_{a.variant}.jsonl"
    mean_pt = {}
    t1 = time.perf_counter()
    with open(out, "w") as f:
        for n, s in enumerate(states):
            res = wm.score(state_in(s), [c["text"] for c in s["cands"]])
            ti = LETTERS.index(s["d"]["target"])
            for c, o in enumerate(res):
                mean_pt[(s["sid"], c)] = o["mean_p_target"]
                for i, p in o["beliefs"].items():
                    p = [float(x) for x in p[: s["d"]["n_options"]]]
                    z = sum(p)
                    p = [x / z for x in p] if z > 0 else [1.0 / len(p)] * len(p)
                    f.write(json.dumps({"id": f"{s['sid']}|c{c}|a{i}", "probs": p, "p_target": p[ti]}) + "\n")
            if (n + 1) % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {n + 1}/{len(states)} states, {time.perf_counter() - t1:.0f}s", flush=True)
    wall = time.perf_counter() - t1
    n_cand = sum(len(s["cands"]) for s in states)
    info = {"model": str(model_path), "variant": a.variant, "judge": PB.SS.NAME, "judge_rev": PB.SS.REV,
            "timing": {"wall_s": wall, "n_turns": 3 * n_cand, "s_per_turn": wall / max(3 * n_cand, 1), "n_candidates": n_cand,
                       "s_per_candidate": wall / max(n_cand, 1), "gen_tokens": 0, "gen_tokens_per_turn": 0.0, "prompt_tokens": 0,
                       "note": "judge scores each candidate once (stance cache); the honest replies are shared by all 8 candidates"},
            "engine_load_s": load_s, "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                                           capture_output=True, text=True).stdout.strip(),
            "host": platform.node(), "python": sys.executable,
            "weights_sha256": {model_path.name: sha256_file(model_path)}, "ppo_bon_sha256": sha256_file(PB.__file__)[:16]}
    if a.check:
        rows = json.load(open(BON / "rows_lambda.json"))
        key = f"WM_{a.variant}_ptarget"
        diffs = [abs(mean_pt[(r["sid"], int(r["c"]))] - float(r[key])) for r in rows if (r["sid"], int(r["c"])) in mean_pt]
        info["check_vs_rows_lambda"] = {"n": len(diffs), "max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs)}
        print("check vs rows_lambda:", json.dumps(info["check_vs_rows_lambda"]), flush=True)
    json.dump(info, open(out.with_suffix(".timing.json"), "w"), indent=1)
    print(json.dumps(info["timing"]))


if __name__ == "__main__":
    main()
