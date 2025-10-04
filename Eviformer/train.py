"""Training utilities for Eviformer models."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .helpers import get_device, one_hot_embedding
from .losses import relu_evidence

__all__ = ["TrainingConfig", "EpochMetrics", "TrainingHistory", "train_model"]


@dataclass
class TrainingConfig:
    num_epochs: int = 100
    num_classes: int = 4
    uncertainty: bool = False


@dataclass
class EpochMetrics:
    epoch: int
    phase: str
    loss: float
    accuracy: float
    mean_evidence: Optional[float] = None
    mean_evidence_success: Optional[float] = None
    mean_evidence_fail: Optional[float] = None
    mean_uncertainty: Optional[float] = None


@dataclass
class TrainingHistory:
    epochs: List[EpochMetrics]
    best_epoch: int
    best_accuracy: float
    duration_seconds: float


def train_model(
    model: torch.nn.Module,
    dataloaders: Dict[str, DataLoader],
    num_classes: int,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs: int = 100,
    device: Optional[torch.device] = None,
    uncertainty: bool = False,
) -> Tuple[torch.nn.Module, TrainingHistory]:
    """Train a model and return the best-performing state along with metrics."""

    config = TrainingConfig(num_epochs=num_epochs, num_classes=num_classes, uncertainty=uncertainty)
    if device is None:
        device = get_device()
    model = model.to(device)

    since = time.perf_counter()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = -1
    history: List[EpochMetrics] = []

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print("-" * 10)

        for phase in ("train", "val"):
            if phase not in dataloaders:
                continue
            is_train = phase == "train"
            if is_train:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            evidence_sum = 0.0
            success_evidence_sum = 0.0
            fail_evidence_sum = 0.0
            uncertainty_sum = 0.0
            match_sum = 0.0
            sample_count = 0

            for inputs, labels, *_ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    if config.uncertainty:
                        targets = one_hot_embedding(labels, config.num_classes)
                        loss = criterion(
                            outputs,
                            targets.float(),
                            epoch,
                            config.num_classes,
                            10,
                            device,
                        )
                        match = torch.eq(preds, labels).float().unsqueeze(1)
                        evidence = relu_evidence(outputs)
                        alpha = evidence + 1
                        batch_uncertainty = (
                            config.num_classes / torch.sum(alpha, dim=1, keepdim=True)
                        )
                        batch_total_evidence = torch.sum(evidence, dim=1, keepdim=True)

                        evidence_sum += batch_total_evidence.sum().item()
                        success_evidence_sum += (batch_total_evidence * match).sum().item()
                        fail_evidence_sum += (
                            batch_total_evidence * (1.0 - match)
                        ).sum().item()
                        uncertainty_sum += batch_uncertainty.sum().item()
                        match_sum += match.sum().item()
                    else:
                        loss = criterion(outputs, labels)

                    if is_train:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                sample_count += inputs.size(0)

            if scheduler is not None and is_train:
                scheduler.step()

            dataset_size = len(dataloaders[phase].dataset)
            if dataset_size == 0:
                continue
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size

            if config.uncertainty and sample_count:
                mean_evidence = evidence_sum / sample_count
                successes = match_sum
                failures = sample_count - successes
                mean_evidence_success = (
                    success_evidence_sum / successes if successes else None
                )
                mean_evidence_fail = (
                    fail_evidence_sum / failures if failures else None
                )
                mean_uncertainty = uncertainty_sum / sample_count
            else:
                mean_evidence = None
                mean_evidence_success = None
                mean_evidence_fail = None
                mean_uncertainty = None

            history.append(
                EpochMetrics(
                    epoch=epoch,
                    phase=phase,
                    loss=epoch_loss,
                    accuracy=epoch_acc,
                    mean_evidence=mean_evidence,
                    mean_evidence_success=mean_evidence_success,
                    mean_evidence_fail=mean_evidence_fail,
                    mean_uncertainty=mean_uncertainty,
                )
            )

            print(f"{phase.capitalize()} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")
            if not is_train and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch

        print()

    duration = time.perf_counter() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            duration // 60, duration % 60
        )
    )
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    metrics = TrainingHistory(
        epochs=history,
        best_epoch=best_epoch,
        best_accuracy=best_acc,
        duration_seconds=duration,
    )
    return model, metrics
