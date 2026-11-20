"""Model bundle (§1): actor (LoRA), reference (adapter toggle), critic
(separate backbone + value head), world model, real-eval benign model.

[TRAP §1] theta (actor LoRA) and phi (critic) are TWO independent parameter
sets, separate optimizers, separate updates. The critic does NOT share the
actor backbone.

Reference model (§1.1, §7.1): rather than loading a second copy of the 9B, the
reference is the frozen SFT base reachable by disabling the actor's LoRA
adapters. Saves ~one 9B of memory; log-probs are identical to a separate frozen
copy of the SFT checkpoint the adapters were initialized from.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def _dtype(bf16: bool):
    return torch.bfloat16 if bf16 else torch.float16


def load_actor(cfg, bf16: bool = True, adapter_path: Optional[str] = None):
    """Base causal LM + LoRA adapters (theta). Plain-text format, no special
    tokens are added (the WM was fine-tuned on plain 'Message:/Position:')."""
    name = cfg.actor.model_name
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.padding_side = "left"          # left-pad for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # fp32 weights for the actor make the logprob recompute EXACT and
    # batch-composition-invariant. In bf16 the batched vs unbatched logprobs
    # drift ~0.06-0.5 (preflight [A'] fails at 5e-1) because bf16 rounds
    # activations at every layer and the reduction order changes with batching;
    # that same drift perturbs PPO old/new ratios (~clip_eps). fp32 -> ~7e-6.
    # Controlled by actor.fp32 (default False to preserve legacy configs).
    actor_fp32 = getattr(cfg.actor, "fp32", False)
    base = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=(torch.float32 if actor_fp32 else _dtype(bf16)),
        trust_remote_code=True,
        # eager attention: deterministic across batch composition (bf16 SDPA
        # picks different kernels for padded vs unpadded batches).
        attn_implementation="eager",
    )

    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    else:
        lora = LoraConfig(
            r=cfg.actor.lora.r,
            lora_alpha=cfg.actor.lora.alpha,
            lora_dropout=cfg.actor.lora.dropout,
            target_modules=list(cfg.actor.lora.target_modules),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora)
    return model, tok


@contextlib.contextmanager
def as_reference(actor: PeftModel):
    """Context where the actor behaves as the frozen reference (adapters off)."""
    with actor.disable_adapter(), torch.no_grad():
        yield actor


class Critic(nn.Module):
    """Separate backbone + linear value head (phi).

    V(s_t) = value_head( hidden_state[last input token of s_t] )   (§3.3)
    i.e. the moment right before the attacker starts generating.
    """

    def __init__(self, model_name: str, bf16: bool = True):
        super().__init__()
        # bf16=False means FP32 (stable full-finetune), NOT fp16 — full-parameter
        # fp16 finetune diverges to NaN. bf16 (for a ~4B critic) is stabilized by
        # eps=1e-5 AdamW + non-finite-grad skip + fp32 value head + Huber loss.
        dtype = torch.bfloat16 if bf16 else torch.float32
        self.backbone = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        )
        hidden = self.backbone.config.hidden_size
        self.value_head = nn.Linear(hidden, 1, bias=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """input_ids: [B, L] state prompts. Returns V: [B] (float32)."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state                 # [B, L, H]
        # index of the last real token per row
        lengths = attention_mask.sum(dim=1) - 1             # [B]
        idx = lengths.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
        pooled = last_hidden.gather(1, idx).squeeze(1)      # [B, H]
        return self.value_head(pooled.float()).squeeze(-1)  # [B]

    def value(self, tokenizer, prompt_text: str, device) -> float:
        ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            return float(self.forward(ids)[0])


def load_critic(cfg, bf16: bool = True) -> Critic:
    # Critic dtype is chosen by `critic.fp32` (default True), NOT train.bf16:
    #   fp32 (default) — stable for <=1.5B full-finetune (bf16 diverged to NaN).
    #   bf16           — needed to fit a ~4B critic; relies on the trainer's
    #                    eps=1e-5 + non-finite-grad skip + Huber for stability.
    fp32 = getattr(cfg.critic, "fp32", True)
    return Critic(cfg.critic.model_name, bf16=(not fp32))


def load_frozen_lm(model_name: str, bf16: bool = True):
    """A frozen causal LM for the world model / real-eval benign agents."""
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "left"          # left-pad for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=_dtype(bf16), trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok
