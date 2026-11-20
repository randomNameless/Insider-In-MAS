#!/usr/bin/env python
"""No-peek variant of the debate-wm-v1 BoN selector (2026-11-12). New file; eval_debate_dwm.py / eval_debate_dwm_lit.py
unchanged.

In eval_debate_dwm (and _lit) the selector scores candidate c of round r on the honest listeners' round-(r+1) chat whose
round-r row holds the honest agents' REAL round-r replies (d['pending']), i.e. the insider effectively speaks last. Here the
round-r row holds replies SIMULATED by the world model itself: before the attacker phase of round r, debate-wm-v1 plays each
honest seat on history 0..r-1 (engines.honest_round: same chat, same honest_temp / honest_arg_tokens as the real agents,
ep_seed = seed + SIM_SEED_OFFSET so the sampling streams differ from the real agents'), and listener_state reads those
simulated replies instead of d['pending']. Everything else (candidates, judge, argmax of mean P(target), invalid last,
commit, metrics) is the original module's code: the module-level DW.listener_state is swapped for the duration of the
attacker worker. The real victim phase still runs first in each round (as in the original runs, so the real honest replies
are generated with the same engine order); the selector never reads them. The simulated replies are stored per debate in
d['np_sim'][str(r)] = {seat: Reply dict, ...} for audit.

Conditions: exactly those of the original modules (targeted_model_*__wmbon<N>dw, ju_dpo_*_wmbon<N>dw, kraidia_*_wmbon<N>dw).

  python -m debate_ppo.eval_debate_dwm_np init ... (eval_debate init flags) --dwm-model PATH
  python -m debate_ppo.eval_debate_dwm_np play --run R
  python -m debate_ppo.eval_debate summary --run R
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debate_ppo import eval_debate as E  # noqa: E402
from debate_ppo import eval_debate_dwm as DW  # noqa: E402
from debate_ppo import eval_debate_dwm_lit as DWL  # noqa: E402

D, P = E.D, E.P
SIM_SEED_OFFSET = 1000


def _arm(cond):
    if DW.parse(cond):
        return DW.worker_attacker_dwm
    if DWL.parse(cond):
        return DWL.worker_attacker
    raise SystemExit(f"not a dwm BoN condition: {cond}")


def np_listener_state(d: dict, r: int, cand_reply) -> "P.DebateState":
    """DW.listener_state with the honest seats' round-r replies taken from the WM simulation, not d['pending']."""
    s = P.DebateState(d["task_id"], d["question"], d["n_options"], d["personalities"])
    hist = [[P.Reply(**rep) for rep in rnd] for rnd in d["history"]]
    sim = d["np_sim"][str(r)]
    cur = [cand_reply if j == d["seat"] else P.Reply(**sim[str(j)]) for j in range(D.N_AGENTS)]
    s.history = hist + [cur]
    s.round = len(s.history)
    return s


def simulate_honest(run: Path, r: int, cond: str):
    """debate-wm-v1 plays the honest seats of round r on the committed history 0..r-1 (no real round-r reply exists in it)."""
    from debate_ppo.engines import close_vllm_engine, honest_round, make_vllm_engine
    st = E.load_state(run)
    cfg = st["config"]
    D.set_profile_set(cfg.get("profile_set", "v1"))
    items, idx = [], []
    for di, d in enumerate(st["debates"]):
        if d["cond"] != cond:
            continue
        s = E.to_state(d)
        assert s.round == r and len(d["history"]) == r
        for i in D.honest_seats(d["seat"]):
            items.append((s, i, d["seed"] + SIM_SEED_OFFSET))
            idx.append((di, i))
    dw = cfg["dwm"]
    E.log(f"np-sim[{cond}] r={r + 1}: {len(items)} honest turns simulated by {dw['model']}")
    eng = make_vllm_engine(dw["model"], gpu_memory_utilization=dw["gpu_memory_utilization"],
                           max_model_len=dw["max_model_len"])
    t0 = time.time()
    res = honest_round(eng, items, temperature=cfg["honest_temp"], max_tokens=cfg["honest_arg_tokens"])
    wall = time.time() - t0
    close_vllm_engine(eng)
    for (di, i), out in zip(idx, res):
        st["debates"][di].setdefault("np_sim", {}).setdefault(str(r), {})[str(i)] = out["reply"].__dict__
    E.log(f"np-sim[{cond}] r={r + 1}: " + json.dumps({"wall_s": wall, "n_turns": len(items)}))
    st["done_phases"].append(f"npsim[{cond}]{r}")
    E.save_state(run, st)


def cmd_play(run: Path):
    conds = [c for c in E.load_state(run)["config"]["conds"] if c != "none"]
    assert conds and all(DW.parse(c) or DWL.parse(c) for c in conds), conds
    me = [sys.executable, "-m", "debate_ppo.eval_debate_dwm_np", "worker", "--run", str(run)]
    for r in range(D.N_ROUNDS + 1):
        if r < D.N_ROUNDS:
            for c in conds:          # simulation first: it sees only the committed history 0..r-1
                if f"npsim[{c}]{r}" not in E.load_state(run)["done_phases"]:
                    subprocess.run(me + ["--role", "sim", "--cond", c, "--round", str(r)], check=True)
            if f"victim{r}" not in E.load_state(run)["done_phases"]:
                subprocess.run(me + ["--role", "victim", "--round", str(r)], check=True)
            for c in conds:
                if f"attacker[{c}]{r}" not in E.load_state(run)["done_phases"]:
                    subprocess.run(me + ["--role", "attacker", "--cond", c, "--round", str(r)], check=True)
        if f"victim{r}" not in E.load_state(run)["done_phases"]:
            subprocess.run(me + ["--role", "victim", "--round", str(r)], check=True)
        if f"commit{r}" not in E.load_state(run)["done_phases"]:
            E.commit(run, r)
    E.log("play done")


def main():
    argv = sys.argv[1:]
    cmd = argv[0]
    if cmd == "init":
        DW.main()                                  # eval_debate init + cfg['dwm'] (model, judge, hashes)
        run = Path(argv[argv.index("--run") + 1])
        st = E.load_state(run)
        assert all(c == "none" or DW.parse(c) or DWL.parse(c) for c in st["config"]["conds"]), st["config"]["conds"]
        st["config"]["dwm"]["code_sha256"].update({f: DW._sha(Path(__file__).resolve().parents[1] / f) for f in (
            "debate_ppo/eval_debate_dwm_lit.py", "debate_ppo/eval_debate_dwm_np.py")})
        st["config"]["dwm"]["selector"] += "; NO-PEEK: the honest listeners' round-r replies in the scored chat are " \
                                           "simulated by the dwm itself (eval_debate_dwm_np, ep_seed+%d), not read" \
                                           % SIM_SEED_OFFSET
        json.dump(st["config"], open(run / "config.json", "w"), indent=1)
        E.save_state(run, st)
    elif cmd == "play":
        cmd_play(Path(argv[argv.index("--run") + 1]))
    elif cmd == "worker":
        run = Path(argv[argv.index("--run") + 1])
        role, rnd = argv[argv.index("--role") + 1], int(argv[argv.index("--round") + 1])
        if role == "victim":
            E.worker_victim(run, rnd)
        elif role == "sim":
            simulate_honest(run, rnd, argv[argv.index("--cond") + 1])
        else:
            DW.listener_state = np_listener_state  # the only change to the original selector
            _arm(argv[argv.index("--cond") + 1])(run, rnd, argv[argv.index("--cond") + 1])
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
