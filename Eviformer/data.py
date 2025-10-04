"""High level data-loading utilities for the CWRU bearing dataset."""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .SequenceDatasets import BearingSequenceDataset, SequenceSample
from .sequence_aug import Compose, Ensure2d, Normalize, ToFloat32

__all__ = [
    "DatasetSummary",
    "build_dataloaders",
    "discover_npz_files",
    "load_npz_segments",
    "main",
]


FAULT_KEYWORDS: Mapping[str, Tuple[str, ...]] = {
    "normal": ("normal",),
    "ball": ("_b_",),
    "inner_race": ("_ir_",),
    "outer_race": ("_or@",),
}
RPM_PATTERN = re.compile(r"(?P<rpm>\d+)\s*rpm", re.IGNORECASE)
LOCATION_PATTERN = re.compile(r"_(DE|FE|BA)(\d+)?", re.IGNORECASE)

DATA_LOADERS: Dict[str, DataLoader] = {}
FIRST_SAMPLE: Optional[Tuple[np.ndarray, int]] = None


@dataclass(frozen=True)
class DatasetSummary:
    """Metadata describing a prepared dataset."""

    source_root: Path
    label_to_index: Dict[str, int]
    split_sizes: Dict[str, int]
    window_size: int
    step_size: int
    samples_per_label: Dict[str, int]


def discover_npz_files(root: Path) -> List[Path]:
    """Recursively discover ``.npz`` files under ``root``."""

    if not root.exists():
        raise FileNotFoundError(f"Data root {root} does not exist")
    files = sorted(path for path in root.rglob("*.npz") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .npz files were found under {root}")
    return files


def infer_fault_label(filename: str) -> str:
    lower = filename.lower()
    for label, tokens in FAULT_KEYWORDS.items():
        if any(token in lower for token in tokens):
            return label
    raise ValueError(f"Unable to infer fault label from filename: {filename}")


def infer_rpm(path: Path) -> Optional[int]:
    for part in path.parts[::-1]:
        match = RPM_PATTERN.search(part)
        if match:
            return int(match.group("rpm"))
    return None


def infer_sensor_location(filename: str) -> Optional[str]:
    match = LOCATION_PATTERN.search(filename)
    if match:
        return match.group(1).upper()
    return None


def load_npz_segments(
    path: Path,
    window_size: int,
    step_size: Optional[int] = None,
) -> np.ndarray:
    """Load a ``.npz`` file and segment signals into fixed-length windows."""

    step = step_size or window_size
    with np.load(path) as npz_file:
        arrays: List[np.ndarray] = []
        for key in sorted(npz_file.files):
            values = np.asarray(npz_file[key])
            if values.ndim == 0:
                continue
            if values.ndim == 1:
                arrays.append(values[np.newaxis, :])
            else:
                leading_dim = values.shape[0]
                arrays.append(values.reshape(leading_dim, -1))
    if not arrays:
        raise ValueError(f"No usable arrays were found in {path}")
    signals = np.concatenate(arrays, axis=0)
    segments: List[np.ndarray] = []
    for row in np.atleast_2d(signals):
        if row.size < window_size:
            raise ValueError(
                f"Signal length {row.size} in {path} is shorter than window size {window_size}"
            )
        for start in range(0, row.size - window_size + 1, step):
            segments.append(row[start : start + window_size])
    return np.stack(segments)


def build_samples(
    files: Sequence[Path],
    window_size: int,
    step_size: Optional[int],
) -> Tuple[List[SequenceSample], Dict[str, int], Dict[str, int]]:
    """Build ``SequenceSample`` instances from raw files."""

    label_to_index: Dict[str, int] = {}
    samples: List[SequenceSample] = []
    per_label: Dict[str, int] = {}
    for file_path in files:
        fault_label = infer_fault_label(file_path.name)
        label_index = label_to_index.setdefault(fault_label, len(label_to_index))
        rpm = infer_rpm(file_path)
        location = infer_sensor_location(file_path.name)
        segments = load_npz_segments(file_path, window_size, step_size)
        per_label[fault_label] = per_label.get(fault_label, 0) + len(segments)
        for segment_idx, segment in enumerate(segments):
            metadata = {
                "source": file_path,
                "fault_label": fault_label,
                "rpm": rpm,
                "sensor_location": location,
                "segment": segment_idx,
            }
            samples.append(SequenceSample(values=segment, label=label_index, metadata=metadata))
    return samples, label_to_index, per_label


def stratified_split(
    labels: Sequence[int],
    val_split: float,
    test_split: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    labels_array = np.asarray(labels)
    if val_split + test_split == 0:
        return indices, np.array([], dtype=int), np.array([], dtype=int)
    if test_split > 0:
        train_indices, temp_indices, train_labels, temp_labels = train_test_split(
            indices,
            labels_array,
            test_size=val_split + test_split,
            stratify=labels_array,
            random_state=random_state,
        )
        relative_test = test_split / (val_split + test_split)
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=relative_test,
            stratify=temp_labels,
            random_state=random_state,
        )
    else:
        train_indices, val_indices = train_test_split(
            indices,
            labels_array,
            test_size=val_split,
            stratify=labels_array,
            random_state=random_state,
        )
        test_indices = np.array([], dtype=int)
    return train_indices, val_indices, test_indices


def build_dataloaders(
    data_root: Path,
    batch_size: int = 64,
    val_split: float = 0.2,
    test_split: float = 0.0,
    window_size: int = 1024,
    step_size: Optional[int] = None,
    normalization: str = "-1-1",
    num_workers: int = 0,
    random_state: int = 42,
) -> Tuple[Dict[str, DataLoader], DatasetSummary]:
    """Create PyTorch dataloaders for the CWRU dataset."""

    files = discover_npz_files(data_root)
    samples, label_to_index, per_label = build_samples(files, window_size, step_size)
    transform = Compose([Ensure2d(), Normalize(normalization), ToFloat32()])
    dataset = BearingSequenceDataset(samples, transform=transform)
    train_idx, val_idx, test_idx = stratified_split(
        dataset.labels, val_split, test_split, random_state
    )
    dataloaders: Dict[str, DataLoader] = {}
    train_dataset = dataset.subset(train_idx) if len(train_idx) else dataset
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    if len(val_idx):
        val_dataset = dataset.subset(val_idx)
    else:
        val_dataset = dataset.subset(train_idx) if len(train_idx) else dataset
    dataloaders["val"] = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    if len(test_idx):
        test_dataset = dataset.subset(test_idx)
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    summary = DatasetSummary(
        source_root=data_root,
        label_to_index=label_to_index,
        split_sizes={key: len(loader.dataset) for key, loader in dataloaders.items()},
        window_size=window_size,
        step_size=step_size or window_size,
        samples_per_label=per_label,
    )
    if train_idx.size:
        first_dataset = dataset.subset(train_idx)
        example = first_dataset[0]
    else:
        example = dataset[0]
    global DATA_LOADERS, FIRST_SAMPLE
    DATA_LOADERS = dataloaders
    tensor, label = example
    FIRST_SAMPLE = (tensor.detach().cpu().numpy(), int(label))
    return dataloaders, summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare CWRU dataloaders")
    parser.add_argument("--data-root", type=Path, default=os.environ.get("CWRU_DATA_ROOT"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.0)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--normalization", choices=["0-1", "-1-1", "mean-std"], default="-1-1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    if args.data_root is None:
        raise ValueError("`--data-root` must be provided or set via CWRU_DATA_ROOT")
    dataloaders, summary = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        window_size=args.window_size,
        step_size=args.step_size,
        normalization=args.normalization,
        num_workers=args.num_workers,
        random_state=args.random_state,
    )
    print(json.dumps(asdict(summary), indent=2, default=str))
    for split, loader in dataloaders.items():
        print(f"{split}: {len(loader.dataset)} samples")


if __name__ == "__main__":
    main()
