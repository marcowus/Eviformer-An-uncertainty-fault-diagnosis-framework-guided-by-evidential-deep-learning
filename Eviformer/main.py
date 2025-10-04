"""Command line interface for training and evaluating Eviformer models."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import DatasetSummary, build_dataloaders
from .helpers import get_device, set_seed
from .losses import edl_digamma_loss, edl_log_loss, edl_mse_loss
from .MCSwinT import mcswint
from .VIT import vit_middle_patch16
from .lenet import LeNet
from .test import evaluate_dataloaders
from .train import train_model

LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": lambda: edl_mse_loss,
    "log": lambda: edl_log_loss,
    "digamma": lambda: edl_digamma_loss,
}

MODEL_CHOICES = {"mcswint", "vit", "lenet"}


def build_model(name: str, num_classes: int, sequence_length: int, dropout: bool) -> nn.Module:
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


def resolve_loss(loss_name: str, uncertainty: bool):
    loss_name = loss_name.lower()
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    if uncertainty and loss_name == "cross_entropy":
        raise ValueError("Cross entropy cannot be used with uncertainty training")
    if not uncertainty and loss_name != "cross_entropy":
        raise ValueError("Uncertainty losses require --uncertainty to be enabled")
    factory = LOSS_REGISTRY[loss_name]
    loss = factory()
    return loss


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Eviformer models")
    parser.add_argument("mode", choices=["train", "test", "inspect"], help="Execution mode")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CWRU_DATA_ROOT"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.0)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--normalization", choices=["0-1", "-1-1", "mean-std"], default="-1-1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES), default="mcswint")
    parser.add_argument("--dropout", action="store_true", help="Enable dropout for compatible models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--scheduler-step", type=int, default=7)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--uncertainty", action="store_true")
    parser.add_argument("--loss", choices=list(LOSS_REGISTRY.keys()), default="cross_entropy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=Path("./results/model.pt"))
    parser.add_argument(
        "--evaluate-splits",
        nargs="*",
        default=None,
        help="Specific dataset splits to evaluate during testing",
    )
    return parser.parse_args(argv)


def ensure_data_root(data_root: Optional[Path]) -> Path:
    if data_root is None:
        raise ValueError("`--data-root` must be provided or set via CWRU_DATA_ROOT")
    return data_root


def train_mode(args: argparse.Namespace, summary: DatasetSummary, dataloaders: Dict[str, DataLoader]) -> None:
    device = get_device(args.device)
    num_classes = len(summary.label_to_index)
    model = build_model(args.model, num_classes, summary.window_size, args.dropout)
    criterion = resolve_loss(args.loss, args.uncertainty)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )
    model, history = train_model(
        model,
        dataloaders,
        num_classes=num_classes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        uncertainty=args.uncertainty,
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "num_classes": num_classes,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "summary": asdict(summary),
        "history": [vars(metric) for metric in history.epochs],
        "best_epoch": history.best_epoch,
        "best_accuracy": history.best_accuracy,
        "duration_seconds": history.duration_seconds,
        "dropout": args.dropout,
        "loss": args.loss,
        "uncertainty": args.uncertainty,
    }
    torch.save(payload, args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")


def test_mode(args: argparse.Namespace, summary: DatasetSummary, dataloaders: Dict[str, DataLoader]) -> None:
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = checkpoint.get("num_classes", len(summary.label_to_index))
    model_name = checkpoint.get("model", args.model)
    checkpoint_summary = checkpoint.get("summary")
    sequence_length = summary.window_size
    if isinstance(checkpoint_summary, dict):
        sequence_length = checkpoint_summary.get("window_size", sequence_length)
    dropout_flag = checkpoint.get("dropout", args.dropout)
    model = build_model(model_name, num_classes, sequence_length, dropout_flag)
    model.load_state_dict(checkpoint["state_dict"])

    use_uncertainty = checkpoint.get("uncertainty", args.uncertainty)
    splits = args.evaluate_splits or list(dataloaders.keys())
    selected_loaders = {split: dataloaders[split] for split in splits if split in dataloaders}
    if not selected_loaders:
        raise ValueError("No valid dataloader splits selected for evaluation")
    criterion = None
    if not use_uncertainty:
        criterion = nn.CrossEntropyLoss()
    metrics = evaluate_dataloaders(
        model,
        selected_loaders,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        uncertainty=use_uncertainty,
    )
    print(json.dumps({split: vars(result) for split, result in metrics.items()}, indent=2))


def inspect_mode(summary: DatasetSummary) -> None:
    print(json.dumps(asdict(summary), indent=2, default=str))


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    data_root = ensure_data_root(args.data_root)
    dataloaders, summary = build_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        window_size=args.window_size,
        step_size=args.step_size,
        normalization=args.normalization,
        num_workers=args.num_workers,
        random_state=args.seed,
    )

    if args.mode == "inspect":
        inspect_mode(summary)
        return

    if args.mode == "train":
        train_mode(args, summary, dataloaders)
        return

    if args.mode == "test":
        test_mode(args, summary, dataloaders)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
