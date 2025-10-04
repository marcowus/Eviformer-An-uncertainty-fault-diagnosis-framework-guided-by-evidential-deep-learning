"""General helper utilities."""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import scipy.ndimage as nd
import torch

__all__ = ["get_device", "one_hot_embedding", "set_seed", "rotate_img"]


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Return the desired torch device, defaulting to CUDA when available."""

    if preferred is not None:
        device = torch.device(preferred)
        if device.type.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_hot_embedding(labels: torch.Tensor, num_classes: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert integer labels to a one-hot encoded tensor."""

    if device is None:
        device = labels.device
    eye = torch.eye(num_classes, device=device)
    return eye[labels]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rotate_img(x: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a flattened image by ``deg`` degrees without changing shape."""

    return nd.rotate(x.reshape(28, 28), deg, reshape=False).ravel()
