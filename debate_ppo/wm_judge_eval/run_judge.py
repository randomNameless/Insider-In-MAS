#!/usr/bin/env python
"""Run one or more judges (judges.py) with one model (one engine load) over the eval set; writes
predictions + timing per judge.

    python -m debate_ppo.wm_judge_eval.run_judge --model PATH --judges roleplay_gen,roleplay_direct \
        --records runs_local/wm_judge_eval/eval_turns.jsonl --out-prefix runs_local/wm_judge_eval/preds/wm
    -> runs_local/wm_judge_eval/preds/wm__roleplay_gen.jsonl (+ .timing.json), wm__roleplay_direct.jsonl, ...

Timing: the eval set is sent to vLLM as ONE batch (continuous batching, like the honest phase of
eval_debate.py and the WM calls in drollout.py). `s_per_turn` is therefore batched throughput, not
single-request latency; `timing.json` next to the output also records the GPU, engine settings and the
model weights' sha256 (first shard) so results can be matched across machines.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from debate_ppo.wm_judge_eval.judges import JUDGES, run_judge  # noqa: E402


def weights_sha256(model_dir: str) -> dict:
    out = {}
    for p in sorted(Path(model_dir).glob("*.safetensors")):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        out[p.name] = h.hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--judges", required=True, help="comma list of " + ",".join(JUDGES))
    ap.add_argument("--records", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7, help="argument sampling T (= world_model.temperature)")
    ap.add_argument("--max-tokens", type=int, default=160, help="argument cap (= world_model.max_new_tokens)")
    ap.add_argument("--cot-tokens", type=int, default=200)
    ap.add_argument("--no-sha", action="store_true")
    a = ap.parse_args()

    from debate_ppo.engines import make_vllm_engine
    judges = a.judges.split(",")
    assert all(j in JUDGES for j in judges), judges
    recs = [json.loads(l) for l in open(a.records)]
    if a.limit:
        recs = recs[: a.limit]
    t0 = time.perf_counter()
    eng = make_vllm_engine(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_model_len,
                           max_num_seqs=a.max_num_seqs)
    load_s = time.perf_counter() - t0
    sha = None if a.no_sha else weights_sha256(a.model)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    for judge in judges:
        print(f"[{time.strftime('%H:%M:%S')}] {judge} x {len(recs)} ...", flush=True)
        res = run_judge(eng, recs, judge, temperature=a.temperature, max_tokens=a.max_tokens, cot_tokens=a.cot_tokens)
        out = Path(f"{a.out_prefix}__{judge}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r, p in zip(recs, res["preds"]):
                f.write(json.dumps({"id": r["id"], **p}) + "\n")
        info = {"model": a.model, "judge": judge, "records": a.records, "n": len(recs), "engine_load_s": load_s,
                "timing": res["timing"], "gpu": gpu, "host": platform.node(),
                "engine": {"gpu_util": a.gpu_util, "max_model_len": a.max_model_len, "max_num_seqs": a.max_num_seqs,
                           "temperature": a.temperature, "max_tokens": a.max_tokens, "cot_tokens": a.cot_tokens,
                           "gdn_prefill_backend": "triton",
                           "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")},
                "weights_sha256": sha}
        json.dump(info, open(out.with_suffix(".timing.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in info.items() if k != "weights_sha256"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
