"""Turn-PPO consensus-game attacker (v1).

Submodules are importable independently; `load_config` is exposed lazily so
importing the pure-math modules (potential, gae, ppo, reward) does not require
PyYAML/torch to be installed.
"""


def load_config(path):
    from .config import load_config as _lc
    return _lc(path)


__all__ = ["load_config"]
