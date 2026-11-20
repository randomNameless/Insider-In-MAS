#!/usr/bin/env python
"""Make an SFT'd Qwen3.5 text checkpoint servable by vLLM 0.23.

The trainer saves Qwen3_5ForCausalLM (text config, keys model.language_model.*). vLLM routes Qwen3.5
through the multimodal Qwen3_5ForConditionalGeneration path and rejects a text-only config. Export =
original ConditionalGeneration config / tokenizer / processor files + the TRAINED language-model
weights + the ORIGINAL (untouched) vision tower and MTP weights, with a fresh shard index.

    python -m debate_wm.export_vllm --trained runs/wm_sft_v1/best --base /path/to/models/qwen3.5-4b --out models/debate-wm-v1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # 1) config / tokenizer / processor files from the original multimodal checkpoint
    for f in os.listdir(a.base):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json" or f.startswith("."):
            continue
        src = os.path.join(a.base, f)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(a.out, f))

    # 2) trained language-model weights
    trained = {}
    for f in sorted(os.listdir(a.trained)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(a.trained, f), "pt") as h:
                for k in h.keys():
                    trained[k] = h.get_tensor(k)
    assert trained and all(k.startswith("model.language_model.") or k == "lm_head.weight" for k in trained), \
        sorted({k.split('.')[0] + '.' + k.split('.')[1] for k in trained})

    # 3) original weights not produced by the text model (vision tower, MTP head)
    base_idx = json.load(open(os.path.join(a.base, "model.safetensors.index.json")))["weight_map"]
    extra, missing = {}, []
    for k, shard in base_idx.items():
        if k in trained:
            continue
        if k.startswith("model.language_model.") or k == "lm_head.weight":
            missing.append(k)
            continue
        with safe_open(os.path.join(a.base, shard), "pt") as h:
            extra[k] = h.get_tensor(k)
    assert not missing, f"trained checkpoint lacks language weights: {missing[:5]}"

    save_file(trained, os.path.join(a.out, "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(extra, os.path.join(a.out, "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    wm = {k: "model-00001-of-00002.safetensors" for k in trained}
    wm.update({k: "model-00002-of-00002.safetensors" for k in extra})
    json.dump({"metadata": {}, "weight_map": wm}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"),
              indent=1)
    meta = os.path.join(a.trained, "meta.json")
    if os.path.exists(meta):
        shutil.copy(meta, os.path.join(a.out, "sft_meta.json"))
    print(f"exported {len(trained)} trained + {len(extra)} original tensors -> {a.out}")


if __name__ == "__main__":
    main()
