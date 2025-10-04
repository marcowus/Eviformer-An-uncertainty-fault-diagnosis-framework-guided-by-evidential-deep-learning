"""Utility functions for computing time-domain statistics on vibration signals."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from scipy.stats import kurtosis, skew


_DEFAULT_SIGNAL_SIZE = 1024

ArrayLike = Union[np.ndarray, Sequence[float]]


def _ensure_2d(signal: ArrayLike) -> Tuple[np.ndarray, bool]:
    """Return a 2-D float64 array and a flag indicating whether input was 1-D."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"Expected a 1-D or 2-D array, got shape {arr.shape} instead.")


def _sanitize_signal(arr: np.ndarray) -> np.ndarray:
    """Replace NaNs/Infs with finite values and demean the signal."""
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return arr


def compute_basic_stats(
    signal: ArrayLike,
    *,
    expected_length: int | None = _DEFAULT_SIGNAL_SIZE,
    enforce_length: bool = True,
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """Compute canonical time-domain statistics for 1-D or batched signals.

    Parameters
    ----------
    signal:
        Input sequence(s). Supports a 1-D array of shape ``(N,)`` or 2-D array of
        shape ``(B, N)``.
    expected_length:
        When provided, the function validates that every sequence length matches
        ``expected_length``. Set to ``None`` to disable the check.
    enforce_length:
        If ``True`` (default) the function raises a ``ValueError`` when the
        length constraint fails. Otherwise the mismatch is ignored.

    Returns
    -------
    Union[Dict[str, float], List[Dict[str, float]]]
        A dictionary of statistics for single input or a list of dictionaries
        for batched input. Keys include ``mean``, ``std``, ``rms``,
        ``peak_to_peak``, ``crest_factor``, ``kurtosis`` and others.
    """

    arr, squeeze = _ensure_2d(signal)
    if expected_length is not None and enforce_length and arr.shape[1] != expected_length:
        raise ValueError(
            f"Signal length mismatch: expected {expected_length} samples, got {arr.shape[1]}"
        )

    arr = _sanitize_signal(arr)

    stats: List[Dict[str, float]] = []
    for row in arr:
        mean_val = float(np.mean(row))
        abs_mean = float(np.mean(np.abs(row)))
        std_val = float(np.std(row, ddof=0))
        rms = float(np.sqrt(np.mean(np.square(row))))
        peak = float(np.max(row))
        trough = float(np.min(row))
        peak_to_peak = float(peak - trough)
        max_abs = float(np.max(np.abs(row)))
        shape_factor = rms / (abs_mean + 1e-12)
        impulse_factor = max_abs / (abs_mean + 1e-12)
        clearance_factor = max_abs / ((np.mean(np.sqrt(np.abs(row))) ** 2) + 1e-12)
        crest_factor = max_abs / (rms + 1e-12)
        stats.append(
            {
                "length": int(row.shape[0]),
                "mean": mean_val,
                "std": std_val,
                "variance": std_val ** 2,
                "rms": rms,
                "abs_mean": abs_mean,
                "max": peak,
                "min": trough,
                "peak_to_peak": peak_to_peak,
                "max_abs": max_abs,
                "crest_factor": crest_factor,
                "kurtosis": float(kurtosis(row, fisher=True, bias=False)),
                "skewness": float(skew(row, bias=False)),
                "shape_factor": shape_factor,
                "impulse_factor": impulse_factor,
                "clearance_factor": clearance_factor,
            }
        )

    return stats[0] if squeeze else stats


__all__ = ["compute_basic_stats"]
