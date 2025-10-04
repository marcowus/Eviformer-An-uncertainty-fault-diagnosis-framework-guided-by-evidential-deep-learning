"""Evaluation helpers for the refactored Eviformer CLI (v1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from helpers import get_device
from losses import relu_evidence

__all__ = ["EvaluationMetricsV1", "evaluate_model_v1", "evaluate_dataloaders_v1"]


@dataclass
class EvaluationMetricsV1:
    loss: Optional[float]
    accuracy: float
    mean_uncertainty: Optional[float]
    total_samples: int


def evaluate_model_v1(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=None,
    device: Optional[torch.device] = None,
    num_classes: Optional[int] = None,
    uncertainty: bool = False,
) -> EvaluationMetricsV1:
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_uncertainty = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels, *_ = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels).item()
            sample_count += inputs.size(0)

            if uncertainty:
                inferred_classes = num_classes or outputs.shape[-1]
                evidence = relu_evidence(outputs)
                alpha = evidence + 1
                batch_uncertainty = (
                    inferred_classes / torch.sum(alpha, dim=1, keepdim=True)
                )
                total_uncertainty += batch_uncertainty.sum().item()

    accuracy = running_corrects / sample_count if sample_count else 0.0
    loss_value = running_loss / sample_count if sample_count and criterion else None
    mean_uncertainty = (
        total_uncertainty / sample_count if sample_count and uncertainty else None
    )
    return EvaluationMetricsV1(
        loss=loss_value,
        accuracy=accuracy,
        mean_uncertainty=mean_uncertainty,
        total_samples=sample_count,
    )


def evaluate_dataloaders_v1(
    model: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    criterion=None,
    device: Optional[torch.device] = None,
    num_classes: Optional[int] = None,
    uncertainty: bool = False,
) -> Dict[str, EvaluationMetricsV1]:
    metrics: Dict[str, EvaluationMetricsV1] = {}
    for name, loader in dataloaders.items():
        metrics[name] = evaluate_model_v1(
            model,
            loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            uncertainty=uncertainty,
        )
    return metrics
