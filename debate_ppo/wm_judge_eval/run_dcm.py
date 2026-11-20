#!/usr/bin/env python
"""The DCM belief world model (latent_agent_transfer/dcm_analysis, the authors' "belief WM": DCM backbone M3 + Qwen2.5-1.5B
stance judge + residual D ensemble, fit on the same 9B fit/val debates) as a judge on the same eval set.

Interface (dcm_analysis/mmlupro/ppo_bon.PPOBeliefWM.score(state, [candidate])): given the honest seats' ROUND-r replies
(message + stated answer + belief vector), the 4th seat's round-(r-1) message and the 4th seat's round-r message as the
candidate, it predicts each honest seat's belief in round r+1. So it can judge rounds 2-4 only (round 1 has no 4th-seat
message) and, unlike the vLLM judges, it reads the honest agents' same-round replies (public text; the next-round prompt of
the vLLM judges contains the same messages) PLUS a belief vector per honest seat:
  tb : the 9B's TRUE recorded belief vector of round r (hidden information; the selector's upper-bound diagnostic)
  wb : the WM's own tracked belief: round-1 prior = per-persona mass on the stated answer (0.925 / 0.814 / 0.745, from
       debate_ppo/wm_bon.py), then its own prediction for the actual spoken 4th-seat message -> observable inputs only,
       the deployable variant used in the WM-BoN runs.
State construction copies debate_ppo/wm_bon.Selector.state (same keys, same raw messages, same prior).
Output: predictions in the eval format (id -> probs) for rounds 2-4, timing per state (= 3 honest seats).

Judge-size variant (2026-10-29): --judge Qwen/Qwen3-4B --judge-rev <sha> --model-suffix _q3_4b --judge-tag 4b uses the
D ensemble re-fit on Qwen3-4B stance scores (results/abcd_mmlupro_ppo{,_pre}/d_model_q3_4b.pt) and the same judge live; the
judge name/revision reach dcm_analysis/stance_scores.py through DCM_JUDGE / DCM_JUDGE_REV (set here before the import).
Output name <prefix>__dcm<judge-tag>_<variant>_<mode>.jsonl (judge-tag '' = the original 1.5B files).

Run with the dcm_analysis environment (it is NOT the insider conda env):
  cd /path/to/latent_agent_transfer/dcm_analysis && python \
     /path/to/repo/debate_ppo/wm_judge_eval/run_dcm.py --mode wb --variant post ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

LETTERS = "ABCDEFGHIJ"
K = 10
PRIOR_MASS = {"stubborn": 0.925, "suggestible": 0.814, "neutral": 0.745}   # = debate_ppo/wm_bon.PRIOR_MASS
LATENT = Path(os.environ.get("LATENT_DIR", "/path/to/latent_agent_transfer/dcm_analysis"))
REPO = Path(__file__).resolve().parents[2]


def _pad(p):
    p = [float(x) for x in p]
    return p + [0.0] * (K - len(p))


def _prior(persona, answer, n_options):
    m = PRIOR_MASS.get(persona, 0.8)
    if answer not in LETTERS[:n_options]:
        return _pad([1.0 / n_options] * n_options)
    p = [(1.0 - m) / (n_options - 1)] * n_options
    p[LETTERS.index(answer)] = m
    return _pad(p)


def sha256_file(p, limit=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def state_for(d, r, mode, tracked):
    """= wm_bon.Selector.state(d, r) with the recorded round-r replies in place of the pending ones."""
    seat = d["seat"]
    hs = [i for i in range(4) if i != seat]
    honest = {}
    for k, i in enumerate(hs):
        rep = d["history"][r][i]
        probs = _pad(d["beliefs"][r][k]) if mode == "tb" else tracked[r][k]
        h = {"persona": d["personalities"][i], "message": rep["raw"], "answer": rep["answer"], "probs": probs}
        if r >= 1:
            prev = d["history"][r - 1][i]
            h.update({"prev_message": prev["raw"], "prev_answer": prev["answer"],
                      "prev_probs": _pad(d["beliefs"][r - 1][k]) if mode == "tb" else tracked[r - 1][k]})
        honest[i] = h
    return {"question_id": d["task_id"], "round": r + 1, "fourth_seat": seat, "gold": d["gold"], "target": d.get("target"),
            "honest": honest, "fourth_prev_message": d["history"][r - 1][seat]["raw"] if r >= 1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(REPO / "runs_local/wm_judge_eval/eval_turns.jsonl"))
    ap.add_argument("--mode", choices=("wb", "tb"), required=True)
    ap.add_argument("--variant", choices=("post", "pre"), default="post",
                    help="post = results/abcd_mmlupro_ppo/d_model.pt (fit on post-argument beliefs, used by WM-BoN); pre = _pre")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debates")
    ap.add_argument("--judge", default="", help="HF name of the stance judge (default: dcm_analysis default, Qwen/Qwen2.5-1.5B)")
    ap.add_argument("--judge-rev", default="", help="HF snapshot sha of --judge")
    ap.add_argument("--model-suffix", default="", help="d_model<suffix>.pt under results/abcd_<data> (e.g. _q3_4b)")
    ap.add_argument("--judge-tag", default="", help="output name tag: <prefix>__dcm<tag>_<variant>_<mode>.jsonl")
    a = ap.parse_args()

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
    from mmlupro.ppo_bon import PPOBeliefWM
    import mmlupro.ppo_bon as PB
    model_path = LATENT / ("results/abcd_" + data_dir.name) / f"d_model{a.model_suffix}.pt"
    t0 = time.perf_counter()
    wm = PPOBeliefWM(model_path=model_path, data_dir=data_dir)
    load_s = time.perf_counter() - t0

    recs = [json.loads(l) for l in open(a.records)]
    manifest = json.load(open(Path(a.records).with_suffix(".manifest.json")))
    wanted = {}
    for r in recs:
        m = r["meta"]
        wanted.setdefault((m["source"], m["cond"], m["task_id"], m["seed"]), []).append(r["id"])
    debates = []
    for run in manifest["sources"]:
        st = json.load(open(REPO / run / "state.json"))
        for d in st["debates"]:
            if (run, d["cond"], d["task_id"], d["seed"]) in wanted:
                debates.append(d)
    debates.sort(key=lambda d: (d["cond"], d["task_id"], d["seed"]))
    if a.limit:
        debates = debates[: a.limit]
    print(f"{len(debates)} debates, engine load {load_s:.0f}s", flush=True)

    out = Path(f"{a.out_prefix}__dcm{a.judge_tag}_{a.variant}_{a.mode}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    n_states = 0
    t1 = time.perf_counter()
    with open(out, "w") as f:
        for n, d in enumerate(debates):
            seat = d["seat"]
            hs = [i for i in range(4) if i != seat]
            tracked = [[_prior(d["personalities"][i], d["history"][0][i]["answer"], d["n_options"]) for i in hs]] \
                if a.mode == "wb" else None
            for r in range(3):                       # 4th-seat message of round r+1 -> honest beliefs of round r+2
                cand = d["history"][r][seat]["raw"]
                res = wm.score(state_for(d, r, a.mode, tracked), [cand])[0]
                n_states += 1
                if a.mode == "wb":
                    tracked.append([_pad(res["beliefs"][i]) for i in hs])
                for i in hs:
                    p = [float(x) for x in res["beliefs"][i][: d["n_options"]]]
                    z = sum(p)
                    p = [x / z for x in p] if z > 0 else [1.0 / d["n_options"]] * d["n_options"]
                    f.write(json.dumps({"id": f"{d['cond']}|{d['task_id']}|s{d['seed']}|r{r + 2}|a{i}", "probs": p}) + "\n")
            if (n + 1) % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {n + 1}/{len(debates)} debates, {time.perf_counter() - t1:.0f}s", flush=True)
    wall = time.perf_counter() - t1
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    info = {"model": str(model_path), "judge": f"dcm{a.judge_tag}_{a.variant}_{a.mode}", "records": a.records, "n": len(recs),
            "n_debates": len(debates), "engine_load_s": load_s,
            "timing": {"wall_s": wall, "n_turns": 3 * n_states, "s_per_turn": wall / max(3 * n_states, 1),
                       "s_per_state": wall / max(n_states, 1), "gen_tokens": 0, "gen_tokens_per_turn": 0.0,
                       "prompt_tokens": 0, "note": "one state = one 4th-seat message scored for 3 honest seats; the "
                       "stance judge scores each message once (cache), as in WM-BoN"},
            "gpu": gpu, "host": platform.node(), "python": sys.executable,
            "engine": {"mode": a.mode, "variant": a.variant, "data_dir": str(data_dir), "judge": PB.SS.NAME, "judge_rev": PB.SS.REV,
                       "ppo_bon_sha256": sha256_file(PB.__file__)[:16]},
            "weights_sha256": {model_path.name: sha256_file(model_path)}}
    json.dump(info, open(out.with_suffix(".timing.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
