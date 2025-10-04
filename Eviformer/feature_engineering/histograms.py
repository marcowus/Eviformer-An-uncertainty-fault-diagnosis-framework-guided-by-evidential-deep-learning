"""Histogram and spectral feature helpers used by the secondary review pipeline."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import entropy

ArrayLike = Union[np.ndarray, Sequence[float]]


def _ensure_2d(signal: ArrayLike) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"Expected a 1-D or 2-D array, got shape {arr.shape} instead.")


def _compute_fft(row: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    window = np.hanning(row.size)
    fft = np.fft.rfft(row * window)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(row.size, d=1.0 / sampling_rate)
    return freqs, mag


def _describe_spectrum(freqs: np.ndarray, mag: np.ndarray, top_k: int) -> Dict[str, List[float]]:
    if mag.size == 0:
        return {"peak_frequencies": [], "peak_magnitudes": [], "bandwidth": 0.0}

    norm_mag = mag / (np.max(mag) + 1e-12)
    peaks, _ = find_peaks(norm_mag, height=0.05, distance=max(1, mag.size // 200))
    if peaks.size == 0:
        peaks = np.array([np.argmax(norm_mag)])
    order = np.argsort(norm_mag[peaks])[::-1]
    top = peaks[order][:top_k]

    centroid = float(np.sum(freqs * norm_mag) / (np.sum(norm_mag) + 1e-12))
    spread = float(
        np.sqrt(np.sum(((freqs - centroid) ** 2) * norm_mag) / (np.sum(norm_mag) + 1e-12))
    )
    return {
        "peak_frequencies": freqs[top].tolist(),
        "peak_magnitudes": norm_mag[top].tolist(),
        "centroid": centroid,
        "spread": spread,
        "bandwidth": float(freqs[-1] - freqs[0]) if freqs.size else 0.0,
    }


def compute_hist_features(
    time_signal: ArrayLike,
    *,
    sampling_rate: float,
    spectrum: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    n_bins: int = 32,
    top_k: int = 5,
) -> Union[Dict[str, Dict[str, Union[float, List[float]]]], List[Dict[str, Dict[str, Union[float, List[float]]]]]]:
    """Compute histogram-based descriptors for time and frequency domain.

    Parameters
    ----------
    time_signal:
        Input time-domain vibration sequence(s).
    sampling_rate:
        Sampling rate in Hz used to derive the frequency axis when FFT needs to
        be computed.
    spectrum:
        Optional tuple ``(freqs, magnitudes)``. When provided it will be reused
        to avoid recomputing the FFT.
    n_bins:
        Number of bins to use for the time-domain amplitude histogram.
    top_k:
        Number of dominant peaks to include from the spectrum description.
    """

    arr, squeeze = _ensure_2d(time_signal)
    features: List[Dict[str, Dict[str, Union[float, List[float]]]]] = []

    for row in arr:
        hist, bin_edges = np.histogram(row, bins=n_bins, density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        amp_entropy = float(entropy(hist + 1e-12, base=np.e))
        amp_stats = {
            "bin_centers": centers.tolist(),
            "densities": hist.tolist(),
            "entropy": amp_entropy,
            "max_density": float(np.max(hist)) if hist.size else 0.0,
        }

        if spectrum is not None:
            freqs, mag = spectrum
        else:
            freqs, mag = _compute_fft(row, sampling_rate)
        spec_desc = _describe_spectrum(freqs, mag, top_k=top_k)

        freq_hist, freq_edges = np.histogram(mag, bins=n_bins, density=True)
        freq_centers = 0.5 * (freq_edges[:-1] + freq_edges[1:])
        freq_entropy = float(entropy(freq_hist + 1e-12, base=np.e))
        freq_stats = {
            "bin_centers": freq_centers.tolist(),
            "densities": freq_hist.tolist(),
            "entropy": freq_entropy,
            "max_density": float(np.max(freq_hist)) if freq_hist.size else 0.0,
        }
        features.append({"time_hist": amp_stats, "freq_hist": freq_stats, "spectrum": spec_desc})

    return features[0] if squeeze else features


__all__ = ["compute_hist_features"]
