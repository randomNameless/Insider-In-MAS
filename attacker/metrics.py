"""Dashboard metrics (§9). All the segment-level and stability signals the
spec calls out must be logged separately — the composite D_t alone hides
everything (§9)."""
from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional


def explained_variance(y_pred: List[float], y_true: List[float]) -> float:
    """1 - Var(y_true - y_pred) / Var(y_true). <0.2 is the danger sign (§9).

    Non-finite entries (a diverged critic emitting NaN/inf) are dropped so a
    transient blowup shows up as a bad EV instead of crashing the run."""
    pairs = [(p, t) for p, t in zip(y_pred, y_true)
             if math.isfinite(p) and math.isfinite(t)]
    if len(pairs) < 2:
        return float("nan")
    var_y = statistics.pvariance([t for _, t in pairs])
    if var_y == 0:
        return 0.0
    resid = [t - p for p, t in pairs]
    return 1.0 - statistics.pvariance(resid) / var_y


def pctl(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
    return xs[k]


class MetricLogger:
    def __init__(self, use_wandb: bool = False, project: str = "consensus-attacker",
                 config: Optional[dict] = None):
        self.use_wandb = use_wandb
        self._wandb = None
        if use_wandb:
            import wandb  # noqa
            self._wandb = wandb
            wandb.init(project=project, config=config or {})

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        flat = {k: v for k, v in metrics.items() if v is not None}
        if self.use_wandb and self._wandb is not None:
            self._wandb.log(flat, step=step)
        line = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in sorted(flat.items()))
        print(f"[step {step}] {line}", flush=True)


def summarize_ratios(log_ratios: List[float]) -> Dict[str, float]:
    if not log_ratios:
        return {}
    return {
        "log_ratio_p50": pctl(log_ratios, 50),
        "log_ratio_p95": pctl(log_ratios, 95),
        "log_ratio_p99": pctl(log_ratios, 99),
    }
