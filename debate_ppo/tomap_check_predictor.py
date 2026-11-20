#!/usr/bin/env python
"""Fidelity check (2026-10-21): re-predict the 12,502 test examples shipped with HakHan/ToMAP-Predictor
(mlp_prediction_results.jsonl) through our port (baseline_tomap.AttitudePredictor: sentence-transformers bge-m3 + their
MLP, input [statement ; format_turns(turns, 'guesser')]). Their file records their own prediction per example, so the
agreement rate measures how exactly our embedding path reproduces theirs (vLLM embedding server in their code)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from debate_ppo import baseline_tomap as TM  # noqa: E402

LABELS = ["Disagree", "Partly Disagree", "Neutral", "Partly Agree", "Agree"]   # class index = attitude 0..4

pred = TM.AttitudePredictor("/path/to/models/bge-m3", "/path/to/models/tomap-predictor/best_mlp_model.pth")
rows = [json.loads(l) for l in open("/path/to/models/tomap-predictor/mlp_prediction_results.jsonl")]
turn_txt = [TM.format_turns(json.loads(r["turns"]) if isinstance(r["turns"], str) else r["turns"], "guesser") for r in rows]
import torch  # noqa: E402
t_vec = pred._emb(turn_txt)
s_vec = pred._emb([r["statement"] for r in rows])
with torch.no_grad():
    cls = torch.argmax(pred.mlp(torch.cat([s_vec, t_vec], dim=1).float()), dim=1).tolist()
ours = [LABELS[c] for c in cls]
agree = sum(o == r["predicted"] for o, r in zip(ours, rows)) / len(rows)
acc_ours = sum(o == r["ground_truth"] for o, r in zip(ours, rows)) / len(rows)
acc_theirs = sum(r["is_correct"] for r in rows) / len(rows)
print(json.dumps({"n": len(rows), "agreement_with_their_predictions": round(agree, 4),
                  "accuracy_ours": round(acc_ours, 4), "accuracy_theirs": round(acc_theirs, 4)}))
