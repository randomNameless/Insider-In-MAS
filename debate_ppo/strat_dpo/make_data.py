#!/usr/bin/env python
"""Strategy-catalog DPO data (2026-10-23). Five stages, each one model in a fresh process, outputs in --out:

  states    debate states from the 9B fit and val transcripts (wm_collect_v1_{fit,val}, condition `none`), rounds 2 and 3
            (0-indexed r = 1, 2): the attacker takes the 4th (neutral) seat and sees the three honest replies of
            round r-1. Target = targets_9b_fitval.json (same rule as the test targets).      -> states.jsonl
  opponent  base Qwen3-4B writes the catalog's opponent model (JSON) per state, greedy.        -> opponent.jsonl
  persuade  base Qwen3-4B writes one message per state x strategy (27), common prompt + strategy.
            Decoding = the model's generation_config (as the Ju et al. arm).                   -> persuade.jsonl
  score     the debate WM plays the three honest agents' next turn with the message in the attacker seat
            (round r committed = real 9B honest replies + the message), K samples with seeds shared by all
            messages of a state; score = mean_{agent,k} P_WM(target). The original neutral reply of the seat is
            scored too (`__orig__`, reference only, never paired).                             -> scores.jsonl
  pairs     per state, valid messages ranked by score: i-th best vs i-th worst (i < --pairs-per-state) if the
            margin >= --margin. Prompt = the NO-strategy prompt, chat-templated (thinking off) with the persuader's
            tokenizer, so training and evaluation see the same text. Splits = the earlier runs' (2026-10-23):
            fit pairs train, val pairs are the held-out set; test questions are never used here.
                                                         -> preference_data.jsonl, preference_eval.jsonl, pairs_summary.json

The gold answer is used only in scoring summaries (P_WM(gold)) and never enters an attacker prompt.

  python -m debate_ppo.strat_dpo.make_data states   --out R --collect ~/runs/wm_collect_v1_fit/state.json
  python -m debate_ppo.strat_dpo.make_data opponent --out R --model ~/models/qwen3-4b
  python -m debate_ppo.strat_dpo.make_data persuade --out R --model ~/models/qwen3-4b
  python -m debate_ppo.strat_dpo.make_data score    --out R --model ~/models/debate-wm-v1
  python -m debate_ppo.strat_dpo.make_data pairs    --out R --model ~/models/qwen3-4b
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from debate_ppo import baseline_bozdag as BZ  # noqa: E402
from debate_ppo import baseline_ju as JU  # noqa: E402
from debate_ppo import dprotocol as D  # noqa: E402
from debate_ppo.ddata import file_sha256  # noqa: E402
from debate_ppo.strat_dpo import strategies as S  # noqa: E402

P = D.P
ORIG = "__orig__"
PERSUADE_MAX_TOKENS = 400        # ~2x the 120-word limit; a cut message is invalid
WM_TEMPERATURE = 0.7             # = honest agents in training / eval (debate_v6_targeted.yaml)
WM_MAX_TOKENS = 160


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _jsonl(path):
    return [json.loads(l) for l in open(path)]


def _write_jsonl(path, rows, mode="w"):
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def model_sha256(path: str) -> dict:
    """sha256 of every weight file of a model dir. Paths differ between machines (local ~/models, HF cache, Lambda
    /path/to/models), so the weights, not the path, identify the model in config.json."""
    ws = sorted(Path(path).glob("*.safetensors")) or sorted(Path(path).glob("*.bin"))
    assert ws, f"no weight files in {path}"
    return {w.name: file_sha256(str(w)) for w in ws}


def _config(out: Path, stage: str, **kw):
    p = out / "config.json"
    cfg = json.load(open(p)) if p.exists() else {}
    if "model" in kw:
        kw["model_sha256"] = model_sha256(kw["model"])
    if "tokenizer" in kw:
        kw["tokenizer_sha256"] = file_sha256(str(Path(kw["tokenizer"]) / "tokenizer.json"))
    cfg[stage] = {**kw, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(cfg, open(p, "w"), indent=1)


# ---------------------------------------------------------------- states
def build_states(debates, targets, seeds, rounds, n=None, split="fit"):
    """Pure function (tested): `none` debates -> attacker states. history = rounds 0..r (round r is the honest
    agents' real reply set; the seat's round-r reply is replaced by each candidate at scoring time)."""
    rows, tasks_seen = [], []
    for d in debates:
        if d["cond"] != "none" or d["seed"] not in seeds:
            continue
        if n is not None and d["task_id"] not in tasks_seen:
            if len(tasks_seen) >= n:
                continue
            tasks_seen.append(d["task_id"])
        tgt = targets[d["task_id"]]
        assert tgt != d["gold"], (d["task_id"], tgt)
        for r in rounds:
            assert 1 <= r < D.N_ROUNDS
            if len(d["history"]) < r + 1:
                continue
            rows.append({"state_id": f"{d['task_id']}|s{d['seed']}|r{r}", "split": split, "task_id": d["task_id"], "seed": d["seed"],
                         "round": r, "seat": d["seat"], "question": d["question"], "n_options": d["n_options"],
                         "gold": d["gold"], "target": tgt, "personalities": d["personalities"],
                         "history": d["history"][: r + 1]})
    return rows


def peers_of(st: dict):
    prev = st["history"][st["round"] - 1]
    return [(prev[j]["argument"], prev[j]["answer"]) for j in range(D.N_AGENTS) if j != st["seat"]]


def cmd_states(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = json.load(open(a.targets))["targets"]
    rows, shas = [], {}
    for path in a.collect.split(","):
        col = json.load(open(path))
        assert col["config"].get("profile_set", "v1") == "v1", col["config"].get("profile_set")
        split = col["config"]["split"]
        assert split in ("fit", "val"), f"{path}: split {split} (test questions must not enter the DPO data)"
        rows += build_states(col["debates"], targets, [int(x) for x in a.seeds.split(",")],
                             [int(x) for x in a.rounds.split(",")], a.n, split)
        shas[path] = file_sha256(path)
    assert len({r["state_id"] for r in rows}) == len(rows)
    _write_jsonl(out / "states.jsonl", rows)
    by = collections.Counter(r["split"] for r in rows)
    _config(out, "states", collect=a.collect, collect_sha256=shas, targets=a.targets, targets_sha256=file_sha256(a.targets),
            seeds=a.seeds, rounds=a.rounds, n=a.n, n_states=dict(by),
            n_questions={sp: len({r["task_id"] for r in rows if r["split"] == sp}) for sp in by})
    log(f"{dict(by)} states ({len({r['task_id'] for r in rows})} questions) -> {out / 'states.jsonl'}")


# ---------------------------------------------------------------- opponent model
def cmd_opponent(a):
    from debate_ppo.engines import make_vllm_engine
    from vllm import SamplingParams
    from protocol import generation_prompt
    out = Path(a.out)
    states = _jsonl(out / "states.jsonl")
    eng = make_vllm_engine(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len)
    chats = []
    for st in states:
        stem, opts, _ = BZ.split_question(st["question"])
        chats.append(S.opponent_model_chat(stem, opts, S.peer_messages(peers_of(st)), st["target"]))
    sp = SamplingParams(temperature=0.0, max_tokens=S.OM_MAX_TOKENS)
    outs = eng.llm.generate([generation_prompt(eng.tok, c) for c in chats], sp, use_tqdm=False)
    rows = []
    for st, o in zip(states, outs):
        om, ok = S.parse_opponent_model(o.outputs[0].text)
        rows.append({"state_id": st["state_id"], "raw": o.outputs[0].text, "opponent_model": om, "json_ok": ok,
                     "cut": o.outputs[0].finish_reason == "length"})
    _write_jsonl(out / "opponent.jsonl", rows)
    _config(out, "opponent", model=a.model, decoding="greedy", max_tokens=S.OM_MAX_TOKENS)
    log(f"opponent models: {len(rows)}, JSON ok {sum(r['json_ok'] for r in rows) / len(rows):.3f}, "
        f"cut {sum(r['cut'] for r in rows)}")


def prompt_parts(st: dict, om: str):
    stem, opts, _ = BZ.split_question(st["question"])
    return stem, opts, S.peer_messages(peers_of(st)), om, st["target"]


# ---------------------------------------------------------------- persuade
def cmd_persuade(a):
    from debate_ppo.engines import make_vllm_engine
    out = Path(a.out)
    states = _jsonl(out / "states.jsonl")
    oms = {r["state_id"]: r["opponent_model"] for r in _jsonl(out / "opponent.jsonl")}
    rows = []
    for st in states:
        for s in S.STRATEGY_IDS:
            rows.append({"state_id": st["state_id"], "strategy": s, "target": st["target"],
                         "seed": D.stable_seed("strat_dpo", st["state_id"], s)})
    by_id = {st["state_id"]: st for st in states}
    chats = [S.attacker_chat(*prompt_parts(by_id[r["state_id"]], oms[r["state_id"]]), strategy=r["strategy"],
                             max_words=a.max_words) for r in rows]
    eng = make_vllm_engine(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len)
    log(f"persuade: {len(states)} states x {len(S.STRATEGY_IDS)} strategies = {len(rows)} messages, "
        f"decoding {JU.sampling_kwargs(a.model)}")
    texts, cuts = JU.generate(eng, chats, [r["seed"] for r in rows], max_tokens=PERSUADE_MAX_TOKENS)
    for r, t, c in zip(rows, texts, cuts):
        r["raw"], r["cut"] = t, c
        r["message"] = S.clean_message(t)
        r.update(S.check_message(r["message"], r["target"], c, a.max_words))
    _write_jsonl(out / "persuade.jsonl", rows)
    _config(out, "persuade", model=a.model, decoding=JU.sampling_kwargs(a.model), max_tokens=PERSUADE_MAX_TOKENS,
            max_words=a.max_words, strategies=S.STRATEGY_IDS)
    val = collections.Counter(r["strategy"] for r in rows if r["valid"])
    log(f"-> {out / 'persuade.jsonl'}: valid {sum(val.values()) / len(rows):.3f}, cut {sum(cuts)}, "
        f"lowest valid rate {min(val.get(s, 0) for s in S.STRATEGY_IDS) / len(states):.3f}")


# ---------------------------------------------------------------- score
def scoring_state(st: dict, reply: "P.Reply") -> "P.DebateState":
    """History 0..r with the seat's round-r reply replaced; the honest agents are about to play round r+1
    (r+1 == 3 is the final submission)."""
    s = P.DebateState(st["task_id"], st["question"], st["n_options"], st["personalities"])
    s.history = [[P.Reply(**rep) for rep in rnd] for rnd in st["history"]]
    s.history[st["round"]][st["seat"]] = reply
    s.round = len(s.history)
    return s


def cmd_score(a):
    from debate_ppo.engines import honest_round, make_vllm_engine
    D.set_profile_set("v1")
    out = Path(a.out)
    states = {st["state_id"]: st for st in _jsonl(out / "states.jsonl")}
    cands = [r for r in _jsonl(out / "persuade.jsonl") if r["valid"]]
    for st in states.values():
        orig = st["history"][st["round"]][st["seat"]]
        cands.append({"state_id": st["state_id"], "strategy": ORIG, "target": st["target"],
                      "message": orig["argument"], "answer": orig["answer"]})
    done = set()
    if (out / "scores.jsonl").exists():
        done = {(r["state_id"], r["strategy"]) for r in _jsonl(out / "scores.jsonl")}
    todo = [c for c in cands if (c["state_id"], c["strategy"]) not in done]
    log(f"score: {len(cands)} candidates ({len(done)} already scored), K={a.k}, WM {a.model}")
    eng = make_vllm_engine(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len,
                           enforce_eager=True)
    for i in range(0, len(todo), a.chunk):
        chunk = todo[i: i + a.chunk]
        items, owners = [], []
        for ci, c in enumerate(chunk):
            st = states[c["state_id"]]
            ans = c.get("answer", st["target"]) if c["strategy"] == ORIG else st["target"]
            s = scoring_state(st, P.Reply(answer=ans, argument=c["message"], raw=c["message"], valid=True))
            for k in range(a.k):                      # ep_seed shared by all candidates of the state
                for ag in D.honest_seats(st["seat"]):
                    items.append((s, ag, D.stable_seed("strat_dpo_score", st["state_id"], k)))
                    owners.append(ci)
        res = honest_round(eng, items, temperature=WM_TEMPERATURE, max_tokens=WM_MAX_TOKENS)
        per = collections.defaultdict(list)
        for ci, h in zip(owners, res):
            per[ci].append(h)
        rows = []
        for ci, c in enumerate(chunk):
            st = states[c["state_id"]]
            ti, gi = D.LETTERS.index(st["target"]), D.LETTERS.index(st["gold"])
            hs = per[ci]
            rows.append({"state_id": c["state_id"], "strategy": c["strategy"],
                         "score": sum(h["probs"][ti] for h in hs) / len(hs),
                         "pgold": sum(h["probs"][gi] for h in hs) / len(hs),
                         "to_target": sum(h["reply"].answer == st["target"] for h in hs) / len(hs),
                         "ptarget_by_sample": [round(h["probs"][ti], 4) for h in hs],
                         "letter_mass": min(h["letter_mass"] for h in hs)})
        _write_jsonl(out / "scores.jsonl", rows, mode="a")
        log(f"  scored {min(i + a.chunk, len(todo))}/{len(todo)}")
    _config(out, "score", model=a.model, k=a.k, temperature=WM_TEMPERATURE, max_tokens=WM_MAX_TOKENS,
            score="mean over honest agents x K of P_WM(target) at the next turn", profile_set="v1")


# ---------------------------------------------------------------- pairs
def rank_pairs(cands, margin: float, n_pairs: int):
    """cands: [(score, strategy, ...)] of one state. -> [(chosen, rejected)]: i-th best vs i-th worst while the two
    are distinct candidates and the score gap >= margin."""
    srt = sorted(cands, key=lambda c: c[0], reverse=True)
    pairs = []
    for i in range(n_pairs):
        hi, lo = srt[i], srt[len(srt) - 1 - i]
        if i >= len(srt) - 1 - i or hi[0] - lo[0] < margin:
            break
        pairs.append((hi, lo))
    return pairs


def cmd_pairs(a):
    from transformers import AutoTokenizer
    from protocol import generation_prompt
    out = Path(a.out)
    tok = AutoTokenizer.from_pretrained(a.model)
    states = {st["state_id"]: st for st in _jsonl(out / "states.jsonl")}
    oms = {r["state_id"]: r["opponent_model"] for r in _jsonl(out / "opponent.jsonl")}
    msgs = {(r["state_id"], r["strategy"]): r for r in _jsonl(out / "persuade.jsonl")}
    scores = collections.defaultdict(list)
    orig = {}
    for r in _jsonl(out / "scores.jsonl"):
        if r["strategy"] == ORIG:
            orig[r["state_id"]] = r["score"]
        else:
            scores[r["state_id"]].append((r["score"], r["strategy"]))
    train, held = [], []
    chosen_s, rejected_s = collections.Counter(), collections.Counter()
    len_c, len_r, gaps, n_states_paired = [], [], [], 0
    for sid, cs in scores.items():
        st = states[sid]
        ps = rank_pairs(cs, a.margin, a.pairs_per_state)
        if not ps:
            continue
        n_states_paired += 1
        prompt = generation_prompt(tok, S.attacker_chat(*prompt_parts(st, oms[sid]), strategy=None,
                                                        max_words=a.max_words))
        for (sc, s_c), (sr, s_r) in ps:
            mc, mr = msgs[(sid, s_c)], msgs[(sid, s_r)]
            row = {"prompt": prompt, "chosen": mc["message"], "rejected": mr["message"], "state_id": sid,
                   "task_id": st["task_id"], "chosen_strategy": s_c, "rejected_strategy": s_r,
                   "chosen_score": sc, "rejected_score": sr}
            (held if st["split"] == "val" else train).append(row)
            chosen_s[s_c] += 1
            rejected_s[s_r] += 1
            len_c.append(mc["words"])
            len_r.append(mr["words"])
            gaps.append(sc - sr)
    _write_jsonl(out / "preference_data.jsonl", train)
    _write_jsonl(out / "preference_eval.jsonl", held)
    per_strat = collections.defaultdict(list)
    for cs in scores.values():
        for sc, s in cs:
            per_strat[s].append(sc)
    pers = _jsonl(out / "persuade.jsonl")
    summ = {
        "n_states": len(states), "n_states_scored": len(scores), "n_states_paired": n_states_paired,
        "pairs_train": len(train), "pairs_holdout": len(held), "margin": a.margin, "pairs_per_state": a.pairs_per_state,
        "score_gap_mean": statistics.mean(gaps) if gaps else None,
        "words_chosen_mean": statistics.mean(len_c) if len_c else None,
        "words_rejected_mean": statistics.mean(len_r) if len_r else None,
        "orig_score_mean": statistics.mean(orig.values()) if orig else None,
        "by_strategy": {s: {"name": S.STRATEGIES[s][0],
                            "valid_rate": sum(r["valid"] for r in pers if r["strategy"] == s) /
                                          max(1, sum(r["strategy"] == s for r in pers)),
                            "score_mean": statistics.mean(per_strat[s]) if per_strat[s] else None,
                            "chosen": chosen_s[s], "rejected": rejected_s[s], "fabrication": s in S.FABRICATION}
                        for s in S.STRATEGY_IDS},
    }
    fit_scores = collections.defaultdict(list)
    for sid, cs in scores.items():
        if states[sid]["split"] == "fit":
            for sc, s in cs:
                fit_scores[s].append(sc)
    summ["best_strategy_fit"] = max(fit_scores, key=lambda s: statistics.mean(fit_scores[s])) if fit_scores else None
    json.dump(summ, open(out / "pairs_summary.json", "w"), indent=1)
    if summ["best_strategy_fit"]:
        (out / "best_strategy.txt").write_text(summ["best_strategy_fit"] + "\n")   # fixed-strategy eval arm
    _config(out, "pairs", tokenizer=a.model, margin=a.margin, pairs_per_state=a.pairs_per_state,
            holdout="val split", prompt="no-strategy prompt, chat template, thinking off")
    log(f"{len(train)} train / {len(held)} held-out pairs from {n_states_paired}/{len(scores)} states; "
        f"words chosen {summ['words_chosen_mean']} vs rejected {summ['words_rejected_mean']}")
    for s in S.STRATEGY_IDS:
        b = summ["by_strategy"][s]
        log(f"  {s} {b['name']:36s} valid {b['valid_rate']:.2f}  score {b['score_mean'] or 0:.3f}  "
            f"chosen {b['chosen']:5d}  rejected {b['rejected']:5d}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("states", "opponent", "persuade", "score", "pairs"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name == "states":
            p.add_argument("--collect", default="/path/to/runs/wm_collect_v1_fit/state.json,"
                                                "/path/to/runs/wm_collect_v1_val/state.json")
            p.add_argument("--targets", default="/path/to/runs/targets_9b_fitval.json")
            p.add_argument("--seeds", default="0")
            p.add_argument("--rounds", default="1,2", help="0-indexed attacker rounds (1 = round 2)")
            p.add_argument("--n", type=int, default=None, help="first n questions per split (smoke)")
        else:
            p.add_argument("--model", required=True)
        if name in ("opponent", "persuade", "score"):
            p.add_argument("--gpu-util", type=float, default=0.85)
            p.add_argument("--max-model-len", type=int, default=8192)
        if name in ("persuade", "pairs"):
            p.add_argument("--max-words", type=int, default=S.MAX_WORDS)
        if name == "score":
            p.add_argument("--k", type=int, default=2, help="honest samples per agent per message")
            p.add_argument("--chunk", type=int, default=2000, help="messages per vLLM call (resumable)")
        if name == "pairs":
            p.add_argument("--margin", type=float, default=0.05)
            p.add_argument("--pairs-per-state", type=int, default=2)
    a = ap.parse_args()
    {"states": cmd_states, "opponent": cmd_opponent, "persuade": cmd_persuade, "score": cmd_score,
     "pairs": cmd_pairs}[a.cmd](a)


if __name__ == "__main__":
    main()
