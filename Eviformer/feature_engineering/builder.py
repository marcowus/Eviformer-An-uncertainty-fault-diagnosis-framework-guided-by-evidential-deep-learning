"""Feature packet builder aggregating statistics for LLM-based review."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from .arma import fit_arma
from .histograms import compute_hist_features
from .statistics import compute_basic_stats

ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass
class ModelOutputs:
    """Container for the primary classifier outputs used by the reviewer."""

    predicted_label: int
    probabilities: Sequence[float]
    uncertainty: float
    alpha: Optional[Sequence[float]] = None
    true_label: Optional[int] = None
    evidence: Optional[float] = None

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "predicted_label": int(self.predicted_label),
            "probabilities": [float(p) for p in self.probabilities],
            "uncertainty": float(self.uncertainty),
            "alpha": None if self.alpha is None else [float(a) for a in self.alpha],
            "true_label": None if self.true_label is None else int(self.true_label),
            "evidence": None if self.evidence is None else float(self.evidence),
        }


def _compute_fft(signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    window = np.hanning(signal.size)
    fft = np.fft.rfft(signal * window)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate)
    return freqs, mag


def build_feature_packet(
    raw_signal: ArrayLike,
    *,
    sampling_rate: float,
    model_outputs: ModelOutputs | Dict[str, Any],
    caches: Optional[MutableMapping[str, Any]] = None,
    histogram_bins: int = 32,
    top_k_spectral_peaks: int = 5,
) -> Dict[str, Any]:
    """Create a structured feature packet for downstream LLM consumption."""

    arr = np.asarray(raw_signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("build_feature_packet expects a single 1-D signal segment.")

    caches = caches if caches is not None else {}

    if "fft" in caches:
        freqs, mag = caches["fft"]
    else:
        freqs, mag = _compute_fft(arr, sampling_rate)
        caches["fft"] = (freqs, mag)

    basic_stats = compute_basic_stats(arr)
    histogram_features = compute_hist_features(
        arr,
        sampling_rate=sampling_rate,
        spectrum=(freqs, mag),
        n_bins=histogram_bins,
        top_k=top_k_spectral_peaks,
    )
    arma_features = fit_arma(arr)

    if isinstance(model_outputs, ModelOutputs):
        model_dict = model_outputs.to_serializable()
    else:
        model_dict = {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in model_outputs.items()
        }

    packet = {
        "time": basic_stats,
        "histogram": histogram_features,
        "arma": arma_features,
        "model_output": model_dict,
        "sampling_rate": float(sampling_rate),
        "signal_preview": {
            "head": arr[:10].tolist(),
            "tail": arr[-10:].tolist(),
            "rms": basic_stats["rms"],
        },
    }
    return packet


__all__ = ["ModelOutputs", "build_feature_packet"]
