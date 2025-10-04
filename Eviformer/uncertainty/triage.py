"""Heuristics for deciding when to trigger the secondary LLM review."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np


@dataclass
class TriageThresholds:
    """Threshold configuration for the escalation logic."""

    max_probability: float = 0.6
    min_uncertainty: float = 0.25
    min_evidence: float = 30.0
    min_alpha_sum: Optional[float] = None


def _to_numpy(values: Union[Sequence[float], np.ndarray, None]) -> Optional[np.ndarray]:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        return values
    return np.asarray(values, dtype=np.float64)


def _compute_evidence(alpha: Optional[np.ndarray]) -> Optional[float]:
    if alpha is None:
        return None
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.size == 0:
        return None
    return float(np.sum(alpha - 1.0))


def should_escalate(
    *,
    probabilities: Sequence[float] | np.ndarray | None,
    uncertainty: float | None,
    alpha: Sequence[float] | np.ndarray | None,
    thresholds: TriageThresholds,
) -> bool:
    """Return ``True`` when the sample warrants escalation to the LLM."""

    prob = _to_numpy(probabilities)
    max_prob = float(np.max(prob)) if prob is not None and prob.size else None

    if max_prob is not None and max_prob < thresholds.max_probability:
        return True

    if uncertainty is not None and uncertainty > thresholds.min_uncertainty:
        return True

    evidence = _compute_evidence(alpha)
    if evidence is not None and evidence < thresholds.min_evidence:
        return True

    if thresholds.min_alpha_sum is not None and alpha is not None:
        alpha_sum = float(np.sum(alpha))
        if alpha_sum < thresholds.min_alpha_sum:
            return True

    return False


def batch_escalation(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    uncertainties: Sequence[float] | np.ndarray,
    alphas: Sequence[Sequence[float]] | np.ndarray,
    thresholds: TriageThresholds,
) -> List[bool]:
    """Vectorised convenience wrapper for ``should_escalate``."""

    probs = np.asarray(probabilities, dtype=np.float64)
    uncerts = np.asarray(uncertainties, dtype=np.float64)
    alphas_arr = np.asarray(alphas, dtype=np.float64)

    decisions: List[bool] = []
    for idx in range(probs.shape[0]):
        decisions.append(
            should_escalate(
                probabilities=probs[idx],
                uncertainty=float(uncerts[idx]),
                alpha=alphas_arr[idx],
                thresholds=thresholds,
            )
        )
    return decisions


__all__ = ["TriageThresholds", "should_escalate", "batch_escalation"]
