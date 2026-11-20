"""Turn-PPO training loop (§7), batched for H100 throughput.

    0. SFT is done EXTERNALLY (e.g. LlamaFactory); pass the LoRA adapter via
       run_train.py --sft-adapter <dir>. It becomes the initial actor AND (via
       adapter toggle) pi_ref — a fluent, in-format, adversarial anchor.
    1. Critic warmup (actor frozen) until value_explained_variance > 0.5
    2. repeat:
        a. sample N episodes (pi_old) — lockstep BATCHED rollout
        b. recompute old_logp with the training engine (batched)
        c. Phi, r_t, V_phi, GAE
        d. minibatches (batched forward), ONE epoch, per-minibatch KL early stop
        e. discard rollout
        f. every ~10 iters: real eval; checkpoint selection by REAL results
"""
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import List

import torch

from .asserts import check_pbrs_telescoping
from .generation import logprobs_at_batched
from .metrics import MetricLogger, explained_variance, summarize_ratios
from .models import as_reference, load_actor, load_critic, load_frozen_lm
from .ppo import (approx_kl_k3, is_clipped, ppo_turn_loss,
                  segment_weighted_mean, turn_log_ratio)
from .reward import (RewardConfig, diversity_penalty, fluency_penalty,
                     length_penalty, message_similarity)
from .rollout import Episode, assign_gae, rollout_batch
from .world_model import BenignEngine


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


@contextmanager
def _isolated_sampling_seed(seed: int):
    """Run stochastic validation reproducibly without perturbing training RNG."""
    py_state = random.getstate()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        random.seed(seed)
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(py_state)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bf16 = cfg.train.bf16

        # Reproduce LoRA initialization, rollout sampling and turn shuffles from
        # one fresh process to the next. Validation uses its own isolated stream.
        train_seed = int(getattr(cfg.train, "seed", 0))
        random.seed(train_seed)
        torch.manual_seed(train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_seed)

        # a couple of cheap H100-friendly flags
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        adapter = getattr(cfg.actor, "sft_adapter", None)
        self.actor, self.actor_tok = load_actor(cfg, bf16=bf16, adapter_path=adapter)
        self.actor.to(self.device)
        # PPO needs DETERMINISTIC policy forwards: eval() disables dropout so the
        # old/new/ref logprobs agree (preflight [A']) and ratios are not corrupted
        # by dropout noise. LoRA params still train (eval() only gates dropout /
        # batchnorm, it does not freeze weights or disable grad).
        self.actor.eval()
        self.actor_pad = self.actor_tok.pad_token_id
        # needed for gradient checkpointing through a frozen base to LoRA params
        try:
            self.actor.enable_input_require_grads()
        except Exception:
            pass

        self.critic = load_critic(cfg, bf16=bf16).to(self.device)
        from transformers import AutoTokenizer
        self.critic_tok = AutoTokenizer.from_pretrained(
            cfg.critic.model_name, trust_remote_code=True)
        if self.critic_tok.pad_token is None:
            self.critic_tok.pad_token = self.critic_tok.eos_token
        self.critic_tok.padding_side = "right"    # critic pools the last real token

        self._defense = self._build_defense(cfg)
        wm, wm_tok = load_frozen_lm(cfg.world_model.model_name, bf16=bf16)
        self.wm = BenignEngine(
            wm.to(self.device), wm_tok, max_pos=cfg.game.max_pos,
            max_new_tokens=cfg.world_model.max_new_tokens,
            personality=cfg.game.benign_personality, msg_order=cfg.game.msg_order,
            defense=self._defense,
            prompt_mode=getattr(cfg.world_model, "prompt_mode", "legacy"))
        self._eval_engine = None

        self.rcfg = RewardConfig(
            alpha_spread=cfg.reward.alpha_spread,
            eta_early_fail=cfg.reward.eta_early_fail,
            invalid_penalty=cfg.reward.invalid_penalty,
            beta_fluency=getattr(cfg.reward, "beta_fluency", 0.1),
            fluency_threshold=getattr(cfg.reward, "fluency_threshold", -4.0),
            beta_length=getattr(cfg.reward, "beta_length", 0.0),
            target_message_tokens=getattr(cfg.reward, "target_message_tokens", 32),
            truncation_penalty=getattr(cfg.reward, "truncation_penalty", 0.0),
            beta_diversity=getattr(cfg.reward, "beta_diversity", 0.0),
            diversity_threshold=getattr(cfg.reward, "diversity_threshold", 0.5),
        )
        # Qwen3.5 GDN needs actor logprob forwards at B=1 for numerical
        # consistency, but optimizer/KL batches must be larger than one turn.
        # Keep those two concepts separate.
        self.logprob_microbatch_turns = int(getattr(
            cfg.ppo, "logprob_microbatch_turns",
            getattr(cfg.ppo, "minibatch_turns", 1)))
        self.update_batch_turns = int(getattr(
            cfg.ppo, "update_batch_turns",
            getattr(cfg.ppo, "minibatch_turns", 1)))
        if self.logprob_microbatch_turns < 1 or self.update_batch_turns < 1:
            raise ValueError("PPO microbatch/update batch sizes must be positive")
        if self.logprob_microbatch_turns > self.update_batch_turns:
            raise ValueError("logprob_microbatch_turns cannot exceed update_batch_turns")
        # eps=1e-5 (not the default 1e-8): bf16 Adam underflows a 1e-8 eps and can
        # blow updates up to NaN, which is what killed the instruct-model run.
        self.opt_actor = torch.optim.AdamW(
            [p for p in self.actor.parameters() if p.requires_grad],
            lr=cfg.lr.actor, eps=1e-5)
        self.opt_critic = torch.optim.AdamW(
            self.critic.parameters(), lr=cfg.lr.critic, eps=1e-5)
        self._nonfinite_skips = 0

        self.logger = MetricLogger(use_wandb=cfg.log.wandb, project=cfg.log.wandb_project,
                                   config=cfg.to_dict())
        self.best_validation_score = (float("inf"), float("inf"))
        self.validation_seed = int(getattr(cfg.eval, "validation_seed", 7_000_000))
        os.makedirs(cfg.train.out_dir, exist_ok=True)

    # ---- helpers -----------------------------------------------------------
    def _build_defense(self, cfg):
        """Optional benign-side defense (Exp2+). cfg.defense.type: 'wmsr' | 'none'.
        None / absent -> no defense (Exp1)."""
        dcfg = getattr(cfg, "defense", None)
        dtype = getattr(dcfg, "type", None) if dcfg is not None else None
        if dtype in (None, "none"):
            return None
        if dtype == "wmsr":
            from .wmsr import WMSRDefense
            F = int(getattr(dcfg, "F", 1))
            print(f"[defense] W-MSR input filter on benign agents, F={F}")
            return WMSRDefense(F=F)
        if dtype == "cpwbft":
            from .cpwbft import CPWBFTDefense
            margin = float(getattr(dcfg, "margin", 0.0))
            print(f"[defense] CP-WBFT confidence input filter on benign agents, margin={margin}")
            return CPWBFTDefense(margin=margin)
        raise ValueError(f"unknown defense.type: {dtype!r}")

    def _set_checkpointing(self, on: bool):
        """Toggle gradient checkpointing on actor + critic. ON only during the
        update (activation-memory saver); OFF during rollout so KV-cached
        generation keeps working (checkpointing forces use_cache=False)."""
        for m in (self.actor, getattr(self.critic, "backbone", None)):
            if m is None:
                continue
            try:
                if on:
                    m.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})
                else:
                    m.gradient_checkpointing_disable()
            except Exception:
                pass

    def _critic_values(self, prompts: List[str], grad: bool):
        cdev = next(self.critic.parameters()).device
        enc = self.critic_tok(prompts, return_tensors="pt", padding=True)
        ids, attn = enc.input_ids.to(cdev), enc.attention_mask.to(cdev)
        if grad:
            return self.critic(ids, attn)
        with torch.no_grad():
            return self.critic(ids, attn)

    def sample_batch(self, base_seed: int, sample: bool = True) -> List[Episode]:
        seeds = [base_seed + i for i in range(self.cfg.ppo.batch_episodes_N)]
        eps = rollout_batch(
            actor=self.actor, actor_tok=self.actor_tok,
            critic=self.critic, critic_tok=self.critic_tok,
            benign_engine=self.wm, cfg=self.cfg, rcfg=self.rcfg,
            seeds=seeds, sample=sample,
        )
        # old_logp/ref_logp must be cached BEFORE advantages: the fluency reward
        # reads ref_logp, and assign_gae reads the (now fluency-inclusive) reward.
        self._recompute_old_and_ref(eps)
        self._add_fluency_reward(eps)
        self._add_message_regularization(eps)
        for ep in eps:
            assign_gae(ep, self.cfg)
        return eps

    def _add_fluency_reward(self, episodes: List[Episode]) -> None:
        """Per-turn anti-gibberish penalty (needs ref_logp -> called after
        _recompute_old_and_ref, before assign_gae). Reference log-prob of the
        MESSAGE tokens = perplexity signal; reward.fluency_penalty adds the
        rule-based catches."""
        for ep in episodes:
            for tn in ep.turns:
                msg_mask = tn.gen.seg_masks["message"]
                if tn.ref_logp is not None and bool(msg_mask.any()):
                    # clamp to a sane floor: a nan_to_num'd -inf logit would make
                    # this ~-1e4 and blow up the reward/return -> NaN critic.
                    tn.msg_ref_logp = max(-20.0, float(tn.ref_logp[msg_mask].mean()))
                else:
                    tn.msg_ref_logp = 0.0
                tn.reward_fluency = fluency_penalty(
                    tn.gen.message, tn.msg_ref_logp, self.rcfg)

    def _add_message_regularization(self, episodes: List[Episode]) -> None:
        """Penalize overlong/capped messages and same-round phrase collapse."""
        turns = [tn for ep in episodes for tn in ep.turns]
        by_round = {}
        for tn in turns:
            n_tokens = int(tn.gen.seg_masks["message"].sum())
            tn.message_tokens = n_tokens
            tn.message_hit_cap = n_tokens >= int(self.cfg.actor.max_new_tokens)
            tn.reward_length = length_penalty(
                n_tokens, tn.message_hit_cap, self.rcfg)
            by_round.setdefault(tn.t, []).append(tn)
        for peers in by_round.values():
            for tn in peers:
                sims = [message_similarity(tn.gen.message, other.gen.message)
                        for other in peers if other is not tn]
                tn.message_max_similarity = max([0.0, *sims])
                tn.reward_diversity = diversity_penalty(sims, self.rcfg)

    def _recompute_old_and_ref(self, episodes: List[Episode]) -> None:
        """§7.b + §7.1: cache old_logp (pi_old) and ref_logp (adapters off),
        both constant across the update epoch. Batched."""
        turns = [tn for ep in episodes for tn in ep.turns]
        for chunk in _chunks(turns, self.logprob_microbatch_turns):
            ids = [tn.gen.input_ids for tn in chunk]
            vidx = [tn.gen.value_idx for tn in chunk]
            with torch.no_grad():
                old = logprobs_at_batched(self.actor, ids, vidx, self.actor_pad)
            with as_reference(self.actor):
                ref = logprobs_at_batched(self.actor, ids, vidx, self.actor_pad)
            for tn, o, r in zip(chunk, old, ref):
                tn.old_logp = o.detach().cpu()
                tn.ref_logp = r.detach().cpu()

    # ---- critic warmup (§7.1) ---------------------------------------------
    def critic_warmup(self):
        for p in self.actor.parameters():
            p.requires_grad_(False)
        cw = self.cfg.critic_warmup
        N = self.cfg.ppo.batch_episodes_N
        mb = self.logprob_microbatch_turns
        print("[critic warmup] sampling one rollout pool (actor frozen) ...")
        episodes = self.sample_batch(base_seed=10_000, sample=True)
        import math as _math
        turns = [tn for ep in episodes for tn in ep.turns if _math.isfinite(tn.ret)]
        print(f"[critic warmup] pool = {len(turns)} turns; fitting critic ...")
        for step in range(cw.steps):
            random.shuffle(turns)
            preds, targets = [], []
            for chunk in _chunks(turns, mb):
                self.opt_critic.zero_grad()
                v = self._critic_values([tn.prompt_text for tn in chunk], grad=True)
                tgt = torch.tensor([tn.ret for tn in chunk], device=v.device)
                (torch.nn.functional.smooth_l1_loss(v, tgt, reduction="sum") / N).backward()
                gn = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.ppo.grad_norm)
                if not torch.isfinite(gn):
                    self.opt_critic.zero_grad()      # skip: never let a NaN grad poison the critic
                    continue
                self.opt_critic.step()
                preds += v.detach().tolist()
                targets += tgt.tolist()
            ev = explained_variance(preds, targets)
            self.logger.log(-cw.steps + step, {"warmup/critic_ev": ev, "warmup/step": step})
            if ev > cw.min_explained_variance and step >= min(10, cw.steps):
                print(f"[critic warmup] EV={ev:.3f} > {cw.min_explained_variance}, done at step {step}")
                break
        for n, p in self.actor.named_parameters():
            if "lora" in n.lower():
                p.requires_grad_(True)

    # ---- one PPO iteration -------------------------------------------------
    def train_iteration(self, it: int):
        cfg = self.cfg
        # sample_batch now also caches old_logp/ref_logp and adds the fluency reward.
        episodes = self.sample_batch(base_seed=1_000_000 + it * 10_000, sample=True)
        check_pbrs_telescoping(episodes[0], cfg.reward.alpha_spread)

        turns = [tn for ep in episodes for tn in ep.turns]
        # advantage whitening (standard PPO): center + scale so the noisy
        # advantage magnitude doesn't dominate/destabilize the gradient.
        import statistics as _st
        _adv = [tn.advantage for tn in turns]
        _mu = _st.mean(_adv)
        _sd = _st.pstdev(_adv) if len(_adv) > 1 else 1.0
        for tn in turns:
            tn.advantage = (tn.advantage - _mu) / (_sd + 1e-8)
        random.shuffle(turns)
        N = cfg.ppo.batch_episodes_N
        eps_clip = cfg.ppo.clip_eps
        beta = cfg.trust_region.beta_kl_ref
        ent_coef = float(getattr(cfg.trust_region, "ent_coef", 0.01))
        ent_coef_position = float(getattr(
            cfg.trust_region, "ent_coef_position", ent_coef))
        ent_coef_message = float(getattr(
            cfg.trust_region, "ent_coef_message", ent_coef))
        segment_weights = {
            "next_position": float(cfg.ppo.w_next_position),
            "message": float(cfg.ppo.w_message),
        }
        device = self.device

        log_ratios, clip_flags, kls = [], [], []
        D_ns, D_ms, entropies, position_entropies, message_entropies = [], [], [], [], []
        loss_ppos, loss_kl_refs, entropy_bonuses, loss_actor_totals = [], [], [], []
        grad_norms = []
        stopped = False

        self._set_checkpointing(True)   # activation-memory saver for the update
        try:
          for update_chunk in _chunks(turns, self.update_batch_turns):
            self.opt_actor.zero_grad()
            self.opt_critic.zero_grad()
            update_kls = []
            # Backward one numerically safe actor microbatch at a time, but do
            # not step until the whole logical update batch has accumulated.
            for micro_chunk in _chunks(update_chunk, self.logprob_microbatch_turns):
                ids = [tn.gen.input_ids for tn in micro_chunk]
                vidx = [tn.gen.value_idx for tn in micro_chunk]
                new_list, ent_list = logprobs_at_batched(
                    self.actor, ids, vidx, self.actor_pad, return_entropy=True)
                vpred = self._critic_values(
                    [tn.prompt_text for tn in micro_chunk], grad=True)
                micro_loss = new_list[0].new_zeros(())
                for k, tn in enumerate(micro_chunk):
                    new_logp = new_list[k]
                    old_logp = tn.old_logp.to(device)
                    ref_logp = tn.ref_logp.to(device)
                    seg = {kk: vv.to(device) for kk, vv in tn.gen.seg_masks.items()}
                    pm, mm = seg["next_position"], seg["message"]
                    D_t = turn_log_ratio(
                        new_logp, old_logp, seg, weights=segment_weights)
                    L_ppo = ppo_turn_loss(D_t, tn.advantage, eps_clip)
                    kl_ref = segment_weighted_mean(
                        new_logp - ref_logp, seg, segment_weights)
                    token_entropy = ent_list[k]
                    mean_entropy = segment_weighted_mean(
                        token_entropy, seg, segment_weights)
                    present_weight = sum(segment_weights[name] for name, mask in seg.items()
                                         if bool(mask.any()))
                    entropy_bonus = token_entropy.new_zeros(())
                    if pm.any():
                        entropy_bonus = entropy_bonus + (
                            segment_weights["next_position"] / present_weight
                            * ent_coef_position * token_entropy[pm].mean())
                    if mm.any():
                        entropy_bonus = entropy_bonus + (
                            segment_weights["message"] / present_weight
                            * ent_coef_message * token_entropy[mm].mean())
                    loss_ppo = -L_ppo
                    loss_kl_ref = beta * kl_ref
                    loss_actor = loss_ppo + loss_kl_ref - entropy_bonus
                    critic_loss = torch.nn.functional.smooth_l1_loss(
                        vpred[k], vpred[k].new_tensor(float(tn.ret)))
                    micro_loss = micro_loss + (loss_actor + critic_loss) / N
                    with torch.no_grad():
                        log_ratios.append(float(D_t))
                        clip_flags.append(is_clipped(D_t, tn.advantage, eps_clip))
                        update_kls.append(float(approx_kl_k3(D_t)))
                        d = new_logp - old_logp
                        if pm.any():
                            D_ns.append(float(d[pm].mean()))
                            position_entropies.append(float(token_entropy[pm].mean()))
                        if mm.any():
                            D_ms.append(float(d[mm].mean()))
                            message_entropies.append(float(token_entropy[mm].mean()))
                        entropies.append(float(mean_entropy))
                        loss_ppos.append(float(loss_ppo))
                        loss_kl_refs.append(float(loss_kl_ref))
                        entropy_bonuses.append(float(entropy_bonus))
                        loss_actor_totals.append(float(loss_actor))
                micro_loss.backward()

            approx_kl = sum(update_kls) / max(1, len(update_kls))
            kls.append(approx_kl)
            if approx_kl > cfg.trust_region.hard_stop_kl:
                self.opt_actor.zero_grad(); self.opt_critic.zero_grad()
                stopped = True
                break

            # Gradients for this logical update batch were already accumulated
            # by each micro_loss.backward() above. Do not backward a second time.
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in self.actor.parameters() if p.requires_grad], cfg.ppo.grad_norm)
            gnc = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.ppo.grad_norm)
            if not (torch.isfinite(gn) and torch.isfinite(gnc)):
                # non-finite grads -> skip the step so NaNs never reach the weights
                self.opt_actor.zero_grad(); self.opt_critic.zero_grad()
                self._nonfinite_skips += 1
                continue
            grad_norms.append(float(gn))
            self.opt_actor.step()
            self.opt_critic.step()
        finally:
            self._set_checkpointing(False)

        self._log_iteration(it, episodes, log_ratios, clip_flags, kls,
                            D_ns, D_ms, entropies, position_entropies,
                            message_entropies, loss_ppos, loss_kl_refs,
                            entropy_bonuses, loss_actor_totals,
                            grad_norms, stopped)
        return episodes

    def _log_iteration(self, it, episodes, log_ratios, clip_flags, kls,
                       D_ns, D_ms, entropies, position_entropies,
                       message_entropies, loss_ppos, loss_kl_refs,
                       entropy_bonuses, loss_actor_totals,
                       grad_norms, stopped):
        turns = [tn for ep in episodes for tn in ep.turns]
        preds = [tn.value for tn in turns]
        targets = [tn.ret for tn in turns]
        lengths = [ep.length for ep in episodes]
        consensus = [1.0 if ep.consensus_reached else 0.0 for ep in episodes]
        m = {
            "turn_clip_frac": sum(clip_flags) / max(1, len(clip_flags)),
            "approx_kl": sum(kls) / max(1, len(kls)),
            "kl_early_stop": 1.0 if stopped else 0.0,
            "nonfinite_skips": self._nonfinite_skips,
            "grad_norm": sum(grad_norms) / max(1, len(grad_norms)),
            "value_explained_variance": explained_variance(preds, targets),
            "policy_entropy": sum(entropies) / max(1, len(entropies)),
            "position_entropy": sum(position_entropies) / max(1, len(position_entropies)),
            "message_entropy": sum(message_entropies) / max(1, len(message_entropies)),
            "loss_ppo": sum(loss_ppos) / max(1, len(loss_ppos)),
            "loss_kl_ref": sum(loss_kl_refs) / max(1, len(loss_kl_refs)),
            "entropy_bonus": sum(entropy_bonuses) / max(1, len(entropy_bonuses)),
            "loss_actor_total": sum(loss_actor_totals) / max(1, len(loss_actor_totals)),
            "D_n_mean": sum(D_ns) / max(1, len(D_ns)),
            "D_m_mean": sum(D_ms) / max(1, len(D_ms)),
            "phi_0": sum(ep.phi_0 for ep in episodes) / len(episodes),
            "phi_T": sum(ep.phi_T for ep in episodes) / len(episodes),
            "phi_gain": sum(ep.phi_T - ep.phi_0 for ep in episodes) / len(episodes),
            "consensus_rate_wm": sum(consensus) / len(consensus),
            "mean_episode_length": sum(lengths) / len(lengths),
            "invalid_output_rate": sum(0 if tn.valid else 1 for tn in turns) / len(turns),
            # reward composition (§4 "四项分开记录") — mean per turn, so you can
            # see how much each term contributes and catch scale drift.
            "reward_spread": sum(tn.reward_spread for tn in turns) / len(turns),
            "reward_terminal": sum(tn.reward_terminal for tn in turns) / len(turns),
            "reward_validity": sum(tn.reward_validity for tn in turns) / len(turns),
            "reward_fluency": sum(tn.reward_fluency for tn in turns) / len(turns),
            "reward_length": sum(tn.reward_length for tn in turns) / len(turns),
            "reward_diversity": sum(tn.reward_diversity for tn in turns) / len(turns),
            "reward_total": sum(tn.reward for tn in turns) / len(turns),
            # msg_ref_logp -> 0 (less negative) as messages become fluent
            "msg_ref_logp": sum(getattr(tn, "msg_ref_logp", 0.0) for tn in turns) / len(turns),
            "message_token_length": sum(tn.message_tokens for tn in turns) / len(turns),
            "message_cap_hit_rate": sum(tn.message_hit_cap for tn in turns) / len(turns),
            "message_similarity": sum(tn.message_max_similarity for tn in turns) / len(turns),
            "unique_message_ratio": len({" ".join(tn.gen.message.lower().split())
                                         for tn in turns}) / len(turns),
        }
        m.update(summarize_ratios(log_ratios))
        self.logger.log(it, m)

    # ---- real eval (§7.f) --------------------------------------------------
    def _get_eval_engine(self):
        if getattr(self.cfg.eval, "reuse_world_model", False):
            return self.wm
        if self._eval_engine is None:
            m, t = load_frozen_lm(self.cfg.real_eval.model_name, bf16=self.cfg.train.bf16)
            self._eval_engine = BenignEngine(
                m.to(self.device), t, max_pos=self.cfg.game.max_pos,
                max_new_tokens=self.cfg.real_eval.max_new_tokens,
                personality=self.cfg.game.benign_personality,
                msg_order=self.cfg.game.msg_order, defense=self._defense,
                prompt_mode=getattr(self.cfg.real_eval, "prompt_mode", "legacy"))
        return self._eval_engine

    @torch.no_grad()
    def real_eval(self, it: int) -> float:
        engine = self._get_eval_engine()
        n = int(self.cfg.eval.real_eval_episodes)
        eval_batch = int(getattr(
            self.cfg.eval, "validation_batch_episodes",
            self.cfg.ppo.batch_episodes_N))
        if n < 1 or eval_batch < 1:
            raise ValueError("validation episode counts must be positive")
        # Every checkpoint sees the exact same initial states AND the exact same
        # actor/WM token-sampling stream.  The old code changed seeds with `it`,
        # so a lucky 16-game draw could masquerade as the best policy.
        with _isolated_sampling_seed(self.validation_seed):
            eps = []
            validation_seeds = [self.validation_seed + i for i in range(n)]
            for seed_chunk in _chunks(validation_seeds, eval_batch):
                eps.extend(rollout_batch(
                    actor=self.actor, actor_tok=self.actor_tok,
                    critic=self.critic, critic_tok=self.critic_tok,
                    benign_engine=engine, cfg=self.cfg, rcfg=self.rcfg,
                    seeds=seed_chunk, sample=True,
                ))
        rate = sum(1 for ep in eps if ep.consensus_reached) / n
        mean_length = sum(ep.length for ep in eps) / n
        self.logger.log(it, {"consensus_rate_real": rate,
                             "validation_mean_episode_length": mean_length,
                             "validation_episodes": n,
                             "validation_seed": self.validation_seed})
        # Primary objective: lower consensus. Tie-break: longer survival.
        score = (rate, -mean_length)
        if score < self.best_validation_score:
            self.best_validation_score = score
            self.save(os.path.join(self.cfg.train.out_dir, "best"))
            print(f"[eval it {it}] new best fixed-validation consensus_rate={rate:.3f} "
                  f"mean_ep_len={mean_length:.2f} "
                  f"(n={n}, batch={eval_batch}, seed={self.validation_seed}) -> saved")
        return rate

    def save(self, path: str, *, include_critic: bool = True):
        os.makedirs(path, exist_ok=True)
        self.actor.save_pretrained(path)
        self.actor_tok.save_pretrained(path)
        if include_critic:
            torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))

    # ---- driver ------------------------------------------------------------
    def fit(self):
        print("=== critic warmup ===")
        self.critic_warmup()
        print("=== PPO ===")
        checkpoint_every = int(getattr(
            self.cfg.train, "checkpoint_every",
            getattr(self.cfg.eval, "real_eval_every", 20)))
        last_it = -1
        try:
            for it in range(self.cfg.train.iterations):
                last_it = it
                self.train_iteration(it)
                step = it + 1
                if checkpoint_every > 0 and step % checkpoint_every == 0:
                    # Actor-only snapshots are enough for checkpoint sweep and
                    # avoid duplicating a multi-GB critic every few iterations.
                    self.save(os.path.join(self.cfg.train.out_dir,
                                           f"checkpoint-{step:04d}"),
                              include_critic=False)
                    print(f"[checkpoint] step {step}: actor snapshot saved")
                if step % self.cfg.eval.real_eval_every == 0:
                    self.real_eval(it)
        except KeyboardInterrupt:
            step = max(0, last_it + 1)
            path = os.path.join(self.cfg.train.out_dir, f"interrupted-{step:04d}")
            print(f"\n[interrupt] saving recoverable checkpoint to {path} ...")
            self.save(path)
            print("[interrupt] saved; re-raising KeyboardInterrupt")
            raise
        self.save(os.path.join(self.cfg.train.out_dir, "last"))
        print("[done] final actor + critic saved to last/")
