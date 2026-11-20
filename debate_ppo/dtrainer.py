"""Turn-PPO trainer for the debate, subclassing attacker.trainer.Trainer.

Inherited unchanged: critic warmup, train_iteration (segment-weighted clipped PPO, KL-to-ref,
entropy, critic Huber loss, KL early stop, non-finite skip), old/ref log-prob recompute, fluency /
length / diversity regularisers, GAE assignment, save().
Replaced: model set-up (honest agents are a vLLM engine, not an HF BenignEngine), batch sampling
(debate rollout), validation (fixed val pool, lowest honest accuracy subject to attacker validity),
and an optional Qwen3.5-9B monitor every `monitor.every` steps (logged only, never selects).
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import List

import torch

from attacker.metrics import MetricLogger
from attacker.models import load_actor, load_critic
from attacker.reward import RewardConfig
from attacker.rollout import assign_gae
from attacker.trainer import Trainer, _chunks, _isolated_sampling_seed

from . import dprotocol as D
from .ddata import file_sha256, split_digest, splits
from .dreward import DebateRewardConfig, field_leak
from .drollout import EpisodeSpec, debate_metrics, play_debates
from .engines import close_vllm_engine, make_vllm_engine


class DebateTrainer(Trainer):
    def __init__(self, cfg):        # noqa: C901  (does not call Trainer.__init__: no HF WM)
        self.cfg = cfg
        self.device = "cuda"
        train_seed = int(getattr(cfg.train, "seed", 0))
        random.seed(train_seed)
        torch.manual_seed(train_seed)
        torch.cuda.manual_seed_all(train_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        os.makedirs(cfg.train.out_dir, exist_ok=True)

        self.data = splits(cfg.data.tasks)
        # targeted persuader (2026-10-19): one assigned wrong option per fit/val question
        tcfg = getattr(cfg, "targeted", None)
        self.targets = None
        if tcfg is not None and getattr(tcfg, "enabled", False):
            self.targets = json.load(open(tcfg.targets_file))["targets"]
            for sp in ("fit", "val"):
                miss = [t["task_id"] for t in self.data[sp] if t["task_id"] not in self.targets]
                assert not miss, f"{len(miss)} {sp} questions without a target, e.g. {miss[:3]}"
                bad = [t["task_id"] for t in self.data[sp] if self.targets[t["task_id"]] == t["answer"]]
                assert not bad, f"target == gold for {bad[:3]}"

        # vLLM engines FIRST (they size their memory pool from free GPU memory at start-up).
        mon = getattr(cfg, "monitor", None)
        self.monitor_on = bool(mon is not None and getattr(mon, "enabled", False))
        wm = cfg.world_model
        self.honest = make_vllm_engine(
            wm.model_name, gpu_memory_utilization=float(wm.gpu_memory_utilization),
            max_model_len=int(wm.max_model_len), seed=0, enable_sleep_mode=self.monitor_on,
            max_num_seqs=int(getattr(wm, "max_num_seqs", 128)),
            enforce_eager=bool(getattr(wm, "enforce_eager", False)))
        torch.cuda.empty_cache()
        # The vLLM engine lives in its own EngineCore process; in THIS (training) process
        # expandable segments only reduce fragmentation of the actor/critic allocations.
        torch.cuda.memory._set_allocator_settings("expandable_segments:True")

        adapter = getattr(cfg.actor, "sft_adapter", None)
        self.actor, self.actor_tok = load_actor(cfg, bf16=cfg.train.bf16, adapter_path=adapter)
        self.actor.to(self.device)
        self.actor.eval()
        self.actor_pad = self.actor_tok.pad_token_id
        try:
            self.actor.enable_input_require_grads()
        except Exception:
            pass
        self.critic = load_critic(cfg, bf16=cfg.train.bf16).to(self.device)
        from transformers import AutoTokenizer
        self.critic_tok = AutoTokenizer.from_pretrained(cfg.critic.model_name, trust_remote_code=True)
        if self.critic_tok.pad_token is None:
            self.critic_tok.pad_token = self.critic_tok.eos_token
        self.critic_tok.padding_side = "right"
        self._defense = None
        self._eval_engine = None

        r = cfg.reward
        self.rcfg = RewardConfig(
            alpha_spread=r.alpha_spread, eta_early_fail=0.0, invalid_penalty=r.invalid_penalty,
            beta_fluency=r.beta_fluency, fluency_threshold=r.fluency_threshold,
            beta_length=r.beta_length, target_message_tokens=r.target_message_tokens,
            truncation_penalty=r.truncation_penalty, beta_diversity=r.beta_diversity,
            diversity_threshold=r.diversity_threshold)
        self.drcfg = DebateRewardConfig(
            alpha_belief=r.alpha_spread, terminal_wrong=r.terminal_wrong,
            terminal_deadlock=r.terminal_deadlock, terminal_correct=r.terminal_correct,
            terminal_target=float(getattr(r, "terminal_target", 1.0)),
            terminal_other_wrong=float(getattr(r, "terminal_other_wrong", 0.0)),
            terminal_target_deadlock=float(getattr(r, "terminal_target_deadlock", 0.0)))
        self.logprob_microbatch_turns = int(cfg.ppo.logprob_microbatch_turns)
        self.update_batch_turns = int(cfg.ppo.update_batch_turns)
        self.opt_actor = torch.optim.AdamW(
            [p for p in self.actor.parameters() if p.requires_grad], lr=cfg.lr.actor, eps=1e-5)
        self.opt_critic = torch.optim.AdamW(self.critic.parameters(), lr=cfg.lr.critic, eps=1e-5)
        self._nonfinite_skips = 0
        self.logger = MetricLogger(use_wandb=False, config=cfg.to_dict())
        self.best_validation_score = (float("inf"), float("inf"))
        self.best_record = None
        self.validation_seed = int(cfg.eval.validation_seed)
        self._write_provenance()

    def _set_checkpointing(self, on: bool):
        """HF applies gradient checkpointing only in train() mode, and the inherited trainer keeps
        the actor in eval() -> no checkpointing, which OOMs on ~2k-token debate prompts with eager
        fp32 attention. The actor has no dropout (LoRA dropout 0, attention_dropout 0, asserted
        below), so train() during the update is numerically identical to eval()."""
        assert float(self.cfg.actor.lora.dropout) == 0.0
        assert float(getattr(self.actor.config, "attention_dropout", 0.0)) == 0.0
        super()._set_checkpointing(on)
        self.actor.train(on)

    # ---------------------------------------------------------------- provenance
    def provenance(self) -> dict:
        cfg = self.cfg
        return {
            "prompt_version": D.PROMPT_VERSION,
            "actor_base": cfg.actor.model_name,
            "critic": cfg.critic.model_name,
            "world_model": cfg.world_model.model_name,
            "monitor_model": cfg.monitor.model_name if self.monitor_on else None,
            "tasks": cfg.data.tasks, "tasks_sha256": file_sha256(cfg.data.tasks),
            "split_digest": {k: split_digest(v) for k, v in self.data.items()},
            "targets_file": getattr(getattr(cfg, "targeted", None), "targets_file", None)
            if self.targets is not None else None,
            "train_seed": int(cfg.train.seed), "batch_episodes_N": int(cfg.ppo.batch_episodes_N),
            "validation_seed": self.validation_seed,
            "config": cfg.to_dict(),
        }

    def _write_provenance(self):
        with open(os.path.join(self.cfg.train.out_dir, "provenance.json"), "w") as f:
            json.dump(self.provenance(), f, indent=1)

    def _jsonl(self, name: str, rec: dict):
        with open(os.path.join(self.cfg.train.out_dir, name), "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------------------------------------------------------------- sampling
    def _train_specs(self, base_seed: int) -> List[EpisodeSpec]:
        rng = random.Random(base_seed)
        tasks = rng.sample(self.data["fit"], int(self.cfg.ppo.batch_episodes_N))
        return [EpisodeSpec(t, rng.randrange(D.N_AGENTS), base_seed + i, self._target(t))
                for i, t in enumerate(tasks)]

    def _target(self, task):
        return self.targets[task["task_id"]] if self.targets is not None else None

    def _val_specs(self, n=None) -> List[EpisodeSpec]:
        val = self.data["val"][: n or len(self.data["val"])]
        return [EpisodeSpec(t, D.stable_seed("seat", t["task_id"], self.validation_seed) % D.N_AGENTS,
                            self.validation_seed + i, self._target(t)) for i, t in enumerate(val)]

    def _play(self, specs, engine, sample=True):
        return play_debates(actor=self.actor, actor_tok=self.actor_tok, critic=self.critic,
                            critic_tok=self.critic_tok, honest_engine=engine, cfg=self.cfg,
                            rcfg=self.rcfg, drcfg=self.drcfg, specs=specs, sample=sample)

    def sample_batch(self, base_seed: int, sample: bool = True):
        t0 = time.time()
        eps = self._play(self._train_specs(base_seed), self.honest, sample)
        t1 = time.time()
        self._recompute_old_and_ref(eps)
        self._add_fluency_reward(eps)
        self._add_message_regularization(eps)
        for ep in eps:
            assign_gae(ep, self.cfg)
        self._last_timing = {"time_rollout_s": t1 - t0, "time_recompute_s": time.time() - t1}
        return eps

    def _add_fluency_reward(self, episodes) -> None:
        super()._add_fluency_reward(episodes)
        for ep in episodes:
            for tn in ep.turns:
                if field_leak(tn.gen.message):
                    tn.reward_fluency -= self.rcfg.beta_fluency

    # ---------------------------------------------------------------- logging
    def _log_iteration(self, it, episodes, *args):
        super()._log_iteration(it, episodes, *args)
        m = {f"debate/{k}": v for k, v in debate_metrics(episodes).items()}
        m.update(getattr(self, "_last_timing", {}))
        m["field_leak_rate"] = sum(field_leak(tn.gen.message) for ep in episodes for tn in ep.turns) / \
            sum(len(ep.turns) for ep in episodes)
        self.logger.log(it, m)
        self._jsonl("train_metrics.jsonl", {"it": it, **m})
        ep = episodes[0]
        self._jsonl("samples.jsonl", {"it": it, "task_id": ep.task_id, "gold": ep.gold, "seat": ep.seat,
                                      "outcome": ep.outcome, "phis": ep.phis,
                                      "transcript": ep.transcript})

    # ---------------------------------------------------------------- validation + monitor
    @torch.no_grad()
    def _eval_pool(self, specs, engine):
        eps = []
        bs = int(self.cfg.eval.validation_batch_episodes)
        with _isolated_sampling_seed(self.validation_seed):
            for chunk in _chunks(specs, bs):
                eps.extend(self._play(chunk, engine, sample=True))
        return eps

    def real_eval(self, it: int) -> float:
        t0 = time.time()
        specs = self._val_specs()
        eps = self._eval_pool(specs, self.honest)
        m = debate_metrics(eps)
        step = it + 1
        rec = {"step": step, "n": len(eps), "validation_seed": self.validation_seed, **m,
               "time_s": round(time.time() - t0, 1)}
        ok = m["attacker_valid"] >= float(self.cfg.eval.min_attacker_valid)
        score = self._val_score(m)
        rec["eligible"] = ok
        if ok and score < self.best_validation_score:
            self.best_validation_score = score
            self.best_record = rec
            self.save(os.path.join(self.cfg.train.out_dir, "best"), meta=rec)
            rec["new_best"] = True
            print(f"[val step {step}] NEW BEST on_target={m.get('honest_on_target', float('nan')):.4f} "
                  f"honest_acc={m['honest_final_acc']:.4f} "
                  f"maj_wrong={m['majority_wrong']:.4f} valid={m['attacker_valid']:.3f} -> best/")
        if self.monitor_on and (step % int(self.cfg.monitor.every) == 0 or it < 0):
            try:
                rec["monitor"] = self.monitor(eps)
            except Exception as e:      # monitoring must never stop training
                import traceback
                traceback.print_exc()
                rec["monitor"] = {"error": repr(e)[:500]}
                torch.cuda.empty_cache()
        self.logger.log(it, {f"val/{k}": v for k, v in m.items()})
        self._jsonl("val_history.jsonl", rec)
        return m["honest_final_acc"]

    def _val_score(self, m: dict):
        """Lower is better. Untargeted: honest accuracy; targeted: honest answers ON the target."""
        if self.targets is not None:
            return (-m["honest_on_target"], -m["majority_on_target"])
        return (m["honest_final_acc"], -m["majority_wrong"])

    def monitor(self, wm_eps) -> dict:
        """Same first-n val debates with the real Qwen3.5-9B honest agents (logged only)."""
        n = int(self.cfg.monitor.n_questions)
        specs = self._val_specs(n)
        t0 = time.time()
        # WM sleeps (weights to CPU, cache freed); the 9B engine is created for this monitor only and
        # shut down afterwards, so it holds no GPU memory during PPO updates.
        self.honest.llm.sleep(level=1)
        torch.cuda.empty_cache()
        eng9 = None
        try:
            eng9 = make_vllm_engine(
                self.cfg.monitor.model_name,
                gpu_memory_utilization=float(self.cfg.monitor.gpu_memory_utilization),
                max_model_len=int(self.cfg.world_model.max_model_len), seed=0,
                max_num_seqs=int(getattr(self.cfg.world_model, "max_num_seqs", 128)))
            eps9 = self._eval_pool(specs, eng9)
        finally:
            if eng9 is not None:
                close_vllm_engine(eng9)
            torch.cuda.empty_cache()
            self.honest.llm.wake_up()
        m9 = debate_metrics(eps9)
        mw = debate_metrics(wm_eps[:n])
        out = {"n": n, "wm": mw, "real9b": m9,
               "gap_honest_acc_9b_minus_wm": m9["honest_final_acc"] - mw["honest_final_acc"],
               "gap_majority_wrong_9b_minus_wm": m9["majority_wrong"] - mw["majority_wrong"],
               "time_s": round(time.time() - t0, 1)}
        print(f"[monitor] n={n} honest_acc WM={mw['honest_final_acc']:.3f} 9B={m9['honest_final_acc']:.3f} "
              f"maj_wrong WM={mw['majority_wrong']:.3f} 9B={m9['majority_wrong']:.3f}")
        return out

    def save(self, path: str, *, include_critic: bool = True, meta: dict = None):
        super().save(path, include_critic=include_critic)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"meta": meta, **self.provenance()}, f, indent=1)

    # ---------------------------------------------------------------- driver
    def fit(self, start_it: int = 0):
        """start_it > 0 = resume from an actor-only checkpoint (actor.sft_adapter = checkpoint dir):
        the LoRA weights are restored, the critic is re-warmed, optimizer moments start fresh."""
        cfg = self.cfg
        if start_it:
            # keep the earlier best/: restore the best eligible validation score of this run
            hist = os.path.join(cfg.train.out_dir, "val_history.jsonl")
            if os.path.exists(hist):
                for line in open(hist):
                    r = json.loads(line)
                    if r.get("eligible") and "honest_final_acc" in r:
                        sc = self._val_score(r)
                        if sc < self.best_validation_score:
                            self.best_validation_score, self.best_record = sc, r
            print(f"=== RESUME at iteration {start_it} (best so far {self.best_validation_score}): "
                  f"validation of the loaded adapter ===")
            self._jsonl("val_history.jsonl", {"resume": True, "start_it": start_it,
                                              "adapter": getattr(cfg.actor, "sft_adapter", None)})
        else:
            print("=== validation of the initial policy (RL LoRA = 0, i.e. model_base) ===")
        self.real_eval(start_it - 1)
        print("=== critic warmup ===")
        self.critic_warmup()
        print("=== PPO ===")
        last_it = -1
        try:
            for it in range(start_it, int(cfg.train.iterations)):
                last_it = it
                t0 = time.time()
                self.train_iteration(it)
                step = it + 1
                print(f"[timing] step {step} {time.time() - t0:.1f}s", flush=True)
                if step % int(cfg.train.checkpoint_every) == 0:
                    self.save(os.path.join(cfg.train.out_dir, f"checkpoint-{step:04d}"),
                              include_critic=False, meta={"step": step})
                if step % int(cfg.eval.real_eval_every) == 0:
                    self.real_eval(it)
        except KeyboardInterrupt:
            step = max(0, last_it + 1)
            self.save(os.path.join(cfg.train.out_dir, f"interrupted-{step:04d}"), meta={"step": step})
            raise
        self.save(os.path.join(cfg.train.out_dir, "last"), meta={"step": int(cfg.train.iterations)})
        print(f"[done] last/ saved; best = {self.best_record}")
