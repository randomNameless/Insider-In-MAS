"""Action parsing — plain-text "Message:/Position:" format, matching the real
env (RL_attack_env_qwen.parse_pos_msg). No JSON, no special tokens."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

POSITION_RE = re.compile(r"[Pp]osition\s*:\s*([-+]?\d+)")
MESSAGE_RE = re.compile(r"[Mm]essage\s*:\s*(.+)")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

FALLBACK_MSG = "Let's keep trying to gather."


@dataclass
class ParsedAction:
    next_position: Optional[int]
    message: Optional[str]
    valid: bool


def parse_pos_msg(text: str, fallback_pos: int, max_pos: int = 20) -> ParsedAction:
    """Parse WM/attacker output into (position, message). Mirrors the env's
    parse_pos_msg, but also reports validity (was a well-formed Position found)."""
    text = THINK_RE.sub("", text)
    m = POSITION_RE.search(text)
    valid_pos = False
    if m:
        try:
            pos = max(0, min(max_pos, int(m.group(1))))
            valid_pos = True
        except ValueError:
            pos = fallback_pos
    else:
        pos = fallback_pos
    mm = MESSAGE_RE.search(text)
    if mm:
        msg = mm.group(1).strip().splitlines()[0].strip()
    else:
        msg = ""
    if not msg:
        msg = FALLBACK_MSG
    return ParsedAction(next_position=pos, message=msg, valid=valid_pos)
