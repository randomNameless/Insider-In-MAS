"""World model: frozen LM (opd-mm3) role-playing the benign agents, prompted
EXACTLY as in RL_attack_env_qwen.py (system = BASE_ROLE + personality suffix,
user = ROUND0/MULTIROUND). For this prototype all 3 benign use the SUGGESTIBLE
suffix. Batched free-form generation, parsed with parse_pos_msg."""
from __future__ import annotations

from typing import List

from .game import GameState, clamp_position
from .generation import generate_benign_batched
from .prompts import BENIGN_IDS, build_chat, build_user_prompt


class BenignEngine:
    def __init__(self, model, tokenizer, *, max_pos=20, max_new_tokens=200,
                 personality="suggestible", msg_order="index", temperature=0.7,
                 defense=None, prompt_mode="legacy"):
        self.model = model
        self.tok = tokenizer
        self.max_pos = max_pos
        self.max_new_tokens = max_new_tokens
        # legacy:     system = BASE_ROLE + persona suffix, no history (opd-mm WMs).
        # wm_persona: consensus_wm arm A  = BASE_ROLE + "You are Agent i." + persona
        #             suffix, public history prepended to the user prompt.
        # wm_id:      consensus_wm arm C  = same without the persona suffix.
        if prompt_mode not in ("legacy", "wm_persona", "wm_id"):
            raise ValueError(f"unknown prompt_mode {prompt_mode!r}")
        self.prompt_mode = prompt_mode
        # Fallback personality only; the per-agent personality is read from each
        # GameState (st.personalities[i]) so a mixed combo (stubborn/suggestible/
        # neutral) is realized purely via the system-prompt suffix (Exp1).
        self.personality = personality
        self.msg_order = msg_order
        self.temperature = temperature
        # Optional input-filter defense (Exp2). If set, exposes
        # defense.visible_others(i, state) -> the neighbor indices benign i still
        # sees this round. None = no defense (Exp1, benign see all others).
        self.defense = defense

    def step_batched(self, states: List[GameState], sample: bool = True):
        """Generate benign moves for MANY states at once. Each state contributes
        3 benign queries. Reads the immutable round-start positions/messages
        shared with the attacker. Updates
        benign positions in place; returns per-state list of messages. Does NOT
        touch last_messages (the rollout does, to keep the 1-round lag)."""
        chats, fallbacks, index = [], [], []
        for si, st in enumerate(states):
            for i in BENIGN_IDS:
                # W-MSR input filter (Exp2): only the retained (non-extreme)
                # neighbors are shown to benign i. None -> all others (Exp1).
                vis = self.defense.visible_others(i, st) if self.defense is not None else None
                if self.prompt_mode == "legacy":
                    # per-agent personality suffix (mixed combo in Exp1)
                    chats.append(build_chat(i, st.personalities[i], st.positions,
                                            st.last_messages, st.t, self.msg_order, st.rng,
                                            visible_others=vis))
                else:
                    from consensus_wm.build_sft import history_block, wm_system
                    persona = st.personalities[i] if self.prompt_mode == "wm_persona" else None
                    user = history_block(st.rounds) + build_user_prompt(
                        i, st.positions, st.last_messages, st.t, self.msg_order, st.rng,
                        visible_others=vis)
                    chats.append([{"role": "system", "content": wm_system(i, persona)},
                                  {"role": "user", "content": user}])
                fallbacks.append(st.positions[i])
                index.append((si, i))
        parsed, confs = generate_benign_batched(
            self.model, self.tok, chats, fallbacks,
            max_new_tokens=self.max_new_tokens, sample=sample,
            temperature=self.temperature, max_pos=self.max_pos)
        per_state = [[None, None, None] for _ in states]
        for (si, i), pa, c in zip(index, parsed, confs):
            st = states[si]
            st.positions[i] = clamp_position(pa.next_position, st.cfg)
            st.confidences[i] = float(c)   # for the CP-WBFT defense (Exp3)
            per_state[si][i] = pa.message
        return per_state
