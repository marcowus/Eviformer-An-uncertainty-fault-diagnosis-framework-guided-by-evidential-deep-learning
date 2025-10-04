"""Model and loss factory helpers for the refactored CLI (v1)."""
from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from lenet import LeNet
from losses import edl_digamma_loss, edl_log_loss, edl_mse_loss
from MCSwinT import mcswint
from VIT import vit_middle_patch16

__all__ = [
    "LOSS_REGISTRY_V1",
    "MODEL_CHOICES_V1",
    "build_model_v1",
    "resolve_loss_v1",
]


LOSS_REGISTRY_V1: Dict[str, Callable[[], Callable]] = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": lambda: edl_mse_loss,
    "log": lambda: edl_log_loss,
    "digamma": lambda: edl_digamma_loss,
}

MODEL_CHOICES_V1 = ("mcswint", "vit", "lenet")


def build_model_v1(name: str, num_classes: int, sequence_length: int, dropout: bool) -> nn.Module:
    name = name.lower()
    if name == "mcswint":
        return mcswint(in_channel=1, out_channel=num_classes)
    if name == "vit":
        return vit_middle_patch16(
            data_size=sequence_length,
            in_c=1,
            num_cls=num_classes,
            h_args=[256, 128, 64, 32],
        )
    if name == "lenet":
        return LeNet(dropout=dropout, num_classes=num_classes)
    raise ValueError(f"Unknown model architecture: {name}")


def resolve_loss_v1(loss_name: str, uncertainty: bool):
    key = loss_name.lower()
    if key not in LOSS_REGISTRY_V1:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    if uncertainty and key == "cross_entropy":
        raise ValueError("Cross entropy cannot be used with uncertainty training")
    if not uncertainty and key != "cross_entropy":
        raise ValueError("Uncertainty-aware losses require --uncertainty to be enabled")
    return LOSS_REGISTRY_V1[key]()
