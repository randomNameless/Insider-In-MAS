"""What does the insider say on OOD debates? (2026-11-05) Surface-style stats of attacker arguments per dataset x cond,
from state.json of exp10_nn (ID) and ood_<ds> runs. Regex features only (no model). Writes ~/runs/ood_style/style.md + samples.jsonl.
python debate_wm/ood_style.py"""
import json, re, random, statistics as S
from pathlib import Path

R = Path("/path/to/runs")
SETS = [("MMLU-Pro", "exp10_nn"), ("TQA", "ood_truthfulqa_mc1"), ("GPQA", "ood_gpqa_diamond"), ("LogiQA", "ood_logiqa2"), ("MedQA", "ood_medqa")]
CONDS = {"targeted_4b": "4b", "targeted_model_base": "base", "targeted_model_rl_sd120": "sd120", "targeted_model_rl_sd120__wmbon8dw": "sd120+dw"}
F = {
    "verdict": r"\b(correct|right|best) (answer|choice|option) (is|=)|\banswer is\b|\bfinal (answer|choice)\b",
    "refs_others": r"\b(agent ?\d|previous (agents|round)|earlier (agents|rounds)|others?\b|everyone|consensus|you all|we should)",
    "concede_then_turn": r"\b(while|although|though)\b[^.]{0,200}\b(correct|right|valid|identif)",
    "authority": r"\b(guideline|cdc|who|textbook|standard|established|well[- ]known|studies|research|literature|consensus|board|definition|by definition|scientific)\b",
    "certainty": r"\b(clearly|definitely|certainly|undoubtedly|obviously|must|always|never|definitive)\b",
    "hedge": r"\b(might|may|perhaps|possibly|likely|i think|not sure|seems)\b",
    "reframe_question": r"\b(the question (asks|is asking|intends|tests)|intended answer|trick|wording|key (detail|word|clue)|actually asks|framing)\b",
    "markdown": r"\*\*",
}
RX = {k: re.compile(v, re.I) for k, v in F.items()}
L = re.compile(r"\b(?:option|answer|choice)?\s*\(?([A-J])\)?\b")


def att_msgs(d):
    for r, rnd in enumerate(d["history"]):
        yield r, rnd[d["seat"]]


rows, samples = [], []
for name, tag in SETS:
    for run in ("ours", "dwmbon"):
        p = R / f"{tag}_{run}" / "state.json"
        if not p.exists():
            continue
        for d in json.load(open(p))["debates"]:
            if d["cond"] not in CONDS or d["final"] is None:
                continue
            fin = [m["answer"] for i, m in enumerate(d["final"]) if i != d["seat"]] if len(d["final"]) == 4 else [m["answer"] for m in d["final"]]
            on_t = sum(a == d["target"] for a in fin) >= 2
            for r, m in att_msgs(d):
                a = m["argument"]
                w = a.split()
                sent = [s for s in re.split(r"(?<=[.!?])\s+", a) if s]
                rows.append(dict(ds=name, cond=CONDS[d["cond"]], r=r, words=len(w), on_target=on_t,
                                 target_mentions=len(re.findall(rf"\b{d['target']}\b", a)),
                                 mean_sent_words=len(w) / max(1, len(sent)),
                                 **{k: bool(RX[k].search(a)) for k in RX}))
                samples.append(dict(ds=name, cond=CONDS[d["cond"]], task=d["task_id"], seed=d["seed"], r=r, gold=d["gold"],
                                    target=d["target"], on_target=on_t, question=d["question"][:600], argument=a))

out = R / "ood_style"
keys = ["words", "mean_sent_words", "target_mentions"] + list(F)
lines = ["| dataset | cond | n msgs | " + " | ".join(keys) + " | majority_on_target |", "|" + "---|" * (len(keys) + 4)]
for name, _ in SETS:
    for c in CONDS.values():
        xs = [x for x in rows if x["ds"] == name and x["cond"] == c]
        if not xs:
            continue
        cells = ["%.1f" % S.mean(x[k] for x in xs) if k in ("words", "mean_sent_words", "target_mentions") else "%.2f" % S.mean(x[k] for x in xs) for k in keys]
        lines.append(f"| {name} | {c} | {len(xs)} | " + " | ".join(cells) + " | %.2f |" % S.mean(x["on_target"] for x in xs[::3]))
# within sd120 OOD: feature rate when majority ends on target vs not
lines += ["", "sd120 (all 4 OOD sets pooled): feature rate in debates that end with majority on target vs not", "",
          "| feature | on target | not on target |", "|---|---|---|"]
xs = [x for x in rows if x["cond"] == "sd120" and x["ds"] != "MMLU-Pro"]
for k in keys:
    a = [x[k] for x in xs if x["on_target"]]; b = [x[k] for x in xs if not x["on_target"]]
    lines.append(f"| {k} | %.2f | %.2f |" % (S.mean(a), S.mean(b)))
(out / "style.md").write_text("\n".join(lines) + "\n")
with open(out / "samples.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")
print("\n".join(lines))
