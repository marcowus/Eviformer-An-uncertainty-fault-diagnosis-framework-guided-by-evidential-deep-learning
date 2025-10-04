"""Utility transforms for sequential sensor data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.signal import resample

__all__ = [
    "Compose",
    "Ensure2d",
    "ToFloat32",
    "Normalize",
    "AddGaussian",
    "RandomAddGaussian",
    "RandomScale",
    "RandomStretch",
    "RandomDropout",
]


Transform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Compose:
    """Compose multiple numpy-based transforms."""

    transforms: Sequence[Transform]

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        array = np.asarray(seq)
        for transform in self.transforms:
            array = transform(array)
        return array


class Ensure2d:
    """Ensure that the incoming signal is 2D with shape (channels, length)."""

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        array = np.asarray(seq)
        if array.ndim == 1:
            array = array[np.newaxis, :]
        elif array.ndim > 2:
            array = array.reshape(array.shape[0], -1)
        if array.shape[0] > array.shape[-1]:
            array = array.T
        return array


class ToFloat32:
    """Cast the incoming array to ``float32`` for PyTorch compatibility."""

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        return np.asarray(seq, dtype=np.float32)


class Normalize:
    """Normalize an array using one of a few common schemes."""

    def __init__(self, mode: str = "-1-1") -> None:
        supported = {"0-1", "-1-1", "mean-std"}
        if mode not in supported:
            raise ValueError(f"Unsupported normalization mode: {mode!r}")
        self.mode = mode

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        array = np.asarray(seq, dtype=np.float32)
        if self.mode == "0-1":
            minimum = np.min(array)
            maximum = np.max(array)
            denom = maximum - minimum
            if denom == 0:
                return np.zeros_like(array)
            return (array - minimum) / denom
        if self.mode == "-1-1":
            minimum = np.min(array)
            maximum = np.max(array)
            denom = maximum - minimum
            if denom == 0:
                return np.zeros_like(array)
            return 2 * (array - minimum) / denom - 1
        mean = np.mean(array)
        std = np.std(array)
        if std == 0:
            return np.zeros_like(array)
        return (array - mean) / std


class AddGaussian:
    """Additive Gaussian noise transform."""

    def __init__(self, sigma: float = 0.01) -> None:
        self.sigma = sigma

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        noise = np.random.normal(loc=0.0, scale=self.sigma, size=seq.shape)
        return np.asarray(seq) + noise


class RandomAddGaussian(AddGaussian):
    """Randomly apply additive Gaussian noise with probability 0.5."""

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        if np.random.randint(2):
            return np.asarray(seq)
        return super().__call__(seq)


class RandomScale:
    """Randomly scale each channel with a Gaussian-distributed factor."""

    def __init__(self, sigma: float = 0.01) -> None:
        self.sigma = sigma

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        array = np.asarray(seq)
        if np.random.randint(2):
            return array
        scale_factor = np.random.normal(loc=1.0, scale=self.sigma, size=(array.shape[0], 1))
        return array * scale_factor


class RandomStretch:
    """Randomly stretch or compress a signal along the time dimension."""

    def __init__(self, sigma: float = 0.3) -> None:
        self.sigma = sigma

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        array = np.asarray(seq)
        if np.random.randint(2):
            return array
        length = array.shape[-1]
        new_length = int(length * (1 + (np.random.rand() - 0.5) * self.sigma))
        stretched = np.zeros_like(array)
        for channel in range(array.shape[0]):
            resampled = resample(array[channel], new_length)
            if new_length <= length:
                offset = np.random.randint(0, length - new_length + 1)
                stretched[channel, offset : offset + new_length] = resampled
            else:
                offset = np.random.randint(0, new_length - length + 1)
                stretched[channel] = resampled[offset : offset + length]
        return stretched


class RandomDropout:
    """Randomly zero out a contiguous region of the signal."""

    def __init__(self, drop_len: int = 20) -> None:
        self.drop_len = drop_len

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        array = np.asarray(seq)
        if np.random.randint(2):
            return array
        max_index = max(array.shape[-1] - self.drop_len, 1)
        start = np.random.randint(0, max_index)
        array = array.copy()
        array[..., start : start + self.drop_len] = 0
        return array
