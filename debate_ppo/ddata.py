"""MMLU-Pro fit / val / test split = debate_pilot/data.py (fit 1,441 / val 160 / test 400), with the
tasks.jsonl location configurable so the same split is reproduced on another machine."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from . import PILOT_DIR  # noqa: F401
import data as pilot_data  # noqa: E402  debate_pilot/data.py

EXPECTED = {"fit": 1441, "val": 160, "test": 400}


def splits(tasks_path: str) -> Dict[str, List[dict]]:
    pilot_data.TASKS = Path(tasks_path)
    tasks = pilot_data.load_tasks()
    if tasks and all(t.get("role") == "ood_test" for t in tasks):     # OOD transfer eval file: everything is test
        return {"test": sorted(tasks, key=lambda t: t["task_id"])}
    s = pilot_data.splits()
    got = {k: len(v) for k, v in s.items()}
    assert got == EXPECTED, f"split sizes {got} != {EXPECTED}"
    return s


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_digest(tasks: List[dict]) -> str:
    return hashlib.sha256("|".join(t["task_id"] for t in tasks).encode()).hexdigest()[:16]
