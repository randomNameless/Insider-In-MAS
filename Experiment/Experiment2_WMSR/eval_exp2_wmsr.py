#!/usr/bin/env python
"""Six-condition Experiment 2 eval with the paper-faithful global MSR filter."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent / "Experiment1_Eval"
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from RL_attack_env_qwen import (  # noqa: E402
    AttackEnv, build_system_prompt, ROUND0_PROMPT_TEMPLATE,
    MULTIROUND_PROMPT_TEMPLATE,
)
import eval_attacks_consensus as base_eval  # noqa: E402
from paper_wmsr import retained_neighbor_indices, self_check  # noqa: E402


class PaperGlobalMSREnv(AttackEnv):
    default_f = 1

    def __init__(self, *args, **kwargs):
        self.wmsr_F = int(self.default_f)
        super().__init__(*args, **kwargs)

    def _visible_others(self, i):
        return retained_neighbor_indices(self.pos, i, self.wmsr_F)

    def _build_messages(self, i):
        visible = self._visible_others(i)
        others_pos = [int(self.pos[j]) for j in visible]
        if self.t == 0:
            user = ROUND0_PROMPT_TEMPLATE.format(
                self_pos=int(self.pos[i]), other_positions=others_pos)
        else:
            order = [
                j for j in self._order_others([j for j in range(4) if j != i])
                if j in set(visible)
            ]
            lines = "\n".join(
                f"- Agent {j}: {self.last_messages[j]}" for j in order
            )
            user = MULTIROUND_PROMPT_TEMPLATE.format(
                self_pos=int(self.pos[i]), other_positions=others_pos,
                message_lines=lines,
            )
        return [
            {"role": "system", "content": build_system_prompt(self.personalities[i])},
            {"role": "user", "content": user},
        ]


def _pop_option(name: str, default: int) -> int:
    for idx, value in enumerate(list(sys.argv)):
        if value == name:
            if idx + 1 >= len(sys.argv):
                raise SystemExit(f"{name} requires an integer")
            parsed = int(sys.argv[idx + 1])
            del sys.argv[idx:idx + 2]
            return parsed
        prefix = name + "="
        if value.startswith(prefix):
            parsed = int(value[len(prefix):])
            del sys.argv[idx]
            return parsed
    return default


def _option_value(name: str):
    for idx, value in enumerate(sys.argv):
        if value == name and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def main() -> None:
    self_check()
    f = _pop_option("--wmsr_F", 1)
    if f != 1:
        print(f"[warning] paper experiment uses f=1; requested f={f}", flush=True)
    PaperGlobalMSREnv.default_f = f
    base_eval.AttackEnv = PaperGlobalMSREnv
    dump_path = _option_value("--dump_json")
    print(f"[defense] PAPER GLOBAL MSR: trim exactly f={f} lowest and highest "
          "received neighbor proposals", flush=True)
    base_eval.main()
    if dump_path:
        path = Path(dump_path).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["defense"] = f"paper_global_msr f={f}"
        payload["filter_semantics"] = "sort received neighbors; slice [f:n-f]"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")


if __name__ == "__main__":
    main()

