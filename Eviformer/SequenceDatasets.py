"""Dataset abstractions for sequential bearing data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .sequence_aug import Compose, Ensure2d, ToFloat32

__all__ = ["SequenceSample", "BearingSequenceDataset"]


@dataclass(frozen=True)
class SequenceSample:
    """A single bearing sequence with metadata."""

    values: np.ndarray
    label: int
    metadata: Dict[str, object]


class BearingSequenceDataset(Dataset[Tuple[torch.Tensor, int]]):
    """PyTorch dataset wrapping ``SequenceSample`` records."""

    def __init__(
        self,
        samples: Sequence[SequenceSample],
        transform: Optional[Compose] = None,
        return_metadata: bool = False,
    ) -> None:
        if not samples:
            raise ValueError("`samples` must contain at least one item.")
        self._samples = list(samples)
        self._return_metadata = return_metadata
        self._transform = transform or Compose([Ensure2d(), ToFloat32()])

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._samples)

    def __getitem__(self, index: int):
        sample = self._samples[index]
        values = np.asarray(sample.values)
        values = self._transform(values)
        tensor = torch.from_numpy(values)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if self._return_metadata:
            return tensor, sample.label, sample.metadata
        return tensor, sample.label

    @property
    def labels(self) -> Sequence[int]:
        return [sample.label for sample in self._samples]

    @property
    def metadata(self) -> Sequence[Dict[str, object]]:
        return [sample.metadata for sample in self._samples]

    def subset(self, indices: Iterable[int]) -> "BearingSequenceDataset":
        subset_samples = [self._samples[i] for i in indices]
        return BearingSequenceDataset(
            subset_samples, transform=self._transform, return_metadata=self._return_metadata
        )
