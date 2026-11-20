#!/usr/bin/env python
"""Ju et al. merge_dpo_to_base.py (commit 3c812f37), paths as CLI arguments; logic unchanged (base loaded with
torch_dtype="auto", adapter merged, model + base tokenizer saved). save_pretrained also writes the base model's
generation_config, so the merged persuader decodes like the base one in eval_debate.py."""
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--base", default="/path/to/models/qwen3-4b")
p.add_argument("--adapter", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, a.adapter, torch_dtype="auto").merge_and_unload()
model.save_pretrained(a.out)
AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
print(f"merged -> {a.out}")
