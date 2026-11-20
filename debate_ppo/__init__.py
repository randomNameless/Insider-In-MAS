"""Turn-PPO attacker for the MMLU-Pro debate (Experiment 9).

Reuses the consensus Turn-PPO stack (attacker/trainer, ppo, gae, models, generation, asserts)
and the debate_pilot protocol / vLLM engine. Nothing in attacker/ or debate_pilot/ is modified.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = REPO_ROOT / "debate_pilot"
for p in (str(REPO_ROOT), str(PILOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
