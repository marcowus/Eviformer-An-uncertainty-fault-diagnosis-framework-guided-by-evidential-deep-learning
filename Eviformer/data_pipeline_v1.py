"""Standalone data preparation utilities for the CWRU bearing dataset (v1)."""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sequence_aug import Compose, Ensure2d, Normalize, ToFloat32
from sequence_dataset_v1 import BearingSequenceDatasetV1, SequenceSampleV1

__all__ = [
    "DatasetSummaryV1",
    "build_dataloaders_v1",
    "discover_npz_files_v1",
    "load_npz_segments_v1",
    "split_indices_v1",
    "scan_data_root_v1",
]


FAULT_KEYWORDS: Mapping[str, Tuple[str, ...]] = {
    "normal": ("normal",),
    "ball": ("_b_",),
    "inner_race": ("_ir_",),
    "outer_race": ("_or@",),
}
RPM_PATTERN = re.compile(r"(?P<rpm>\d+)\s*rpm", re.IGNORECASE)
LOCATION_PATTERN = re.compile(r"_(DE|FE|BA)(\d+)?", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetSummaryV1:
    """Lightweight metadata describing a prepared dataset."""

    source_root: Path
    label_to_index: Dict[str, int]
    split_sizes: Dict[str, int]
    window_size: int
    step_size: int
    samples_per_label: Dict[str, int]


class DatasetAssemblyError(RuntimeError):
    """Raised when raw ``.npz`` files cannot be converted into training samples."""


def discover_npz_files_v1(root: Path) -> List[Path]:
    """Recursively discover ``.npz`` files under ``root``."""

    if not root.exists():
        raise FileNotFoundError(f"Data root {root} does not exist")
    files = sorted(path for path in root.rglob("*.npz") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No .npz files were found under {root}")
    return files


def infer_fault_label_v1(filename: str) -> str:
    lower = filename.lower()
    for label, tokens in FAULT_KEYWORDS.items():
        if any(token in lower for token in tokens):
            return label
    raise DatasetAssemblyError(f"Unable to infer fault label from filename: {filename}")


def infer_rpm_v1(path: Path) -> Optional[int]:
    for part in reversed(path.parts):
        match = RPM_PATTERN.search(part)
        if match:
            return int(match.group("rpm"))
    return None


def infer_sensor_location_v1(filename: str) -> Optional[str]:
    match = LOCATION_PATTERN.search(filename)
    if match:
        return match.group(1).upper()
    return None


def load_npz_segments_v1(
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
        raise DatasetAssemblyError(f"No usable arrays were found in {path}")
    signals = np.concatenate(arrays, axis=0)
    segments: List[np.ndarray] = []
    for row in np.atleast_2d(signals):
        if row.size < window_size:
            raise DatasetAssemblyError(
                f"Signal length {row.size} in {path} is shorter than window size {window_size}"
            )
        for start in range(0, row.size - window_size + 1, step):
            segments.append(row[start : start + window_size])
    return np.stack(segments)


def assemble_samples_v1(
    files: Sequence[Path],
    window_size: int,
    step_size: Optional[int],
) -> Tuple[List[SequenceSampleV1], Dict[str, int], Dict[str, int]]:
    label_to_index: Dict[str, int] = {}
    samples: List[SequenceSampleV1] = []
    per_label: Dict[str, int] = {}
    for file_path in files:
        fault_label = infer_fault_label_v1(file_path.name)
        label_index = label_to_index.setdefault(fault_label, len(label_to_index))
        rpm = infer_rpm_v1(file_path)
        location = infer_sensor_location_v1(file_path.name)
        segments = load_npz_segments_v1(file_path, window_size, step_size)
        per_label[fault_label] = per_label.get(fault_label, 0) + len(segments)
        for segment_idx, segment in enumerate(segments):
            metadata = {
                "source": file_path,
                "fault_label": fault_label,
                "rpm": rpm,
                "sensor_location": location,
                "segment": segment_idx,
            }
            samples.append(
                SequenceSampleV1(values=segment, label=label_index, metadata=metadata)
            )
    return samples, label_to_index, per_label


def split_indices_v1(
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


def build_dataloaders_v1(
    data_root: Path,
    batch_size: int = 64,
    val_split: float = 0.2,
    test_split: float = 0.0,
    window_size: int = 1024,
    step_size: Optional[int] = None,
    normalization: str = "-1-1",
    num_workers: int = 0,
    random_state: int = 42,
) -> Tuple[Dict[str, DataLoader], DatasetSummaryV1]:
    files = discover_npz_files_v1(data_root)
    samples, label_to_index, per_label = assemble_samples_v1(files, window_size, step_size)
    transform = Compose([Ensure2d(), Normalize(normalization), ToFloat32()])
    dataset = BearingSequenceDatasetV1(samples, transform=transform)
    train_idx, val_idx, test_idx = split_indices_v1(
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

    summary = DatasetSummaryV1(
        source_root=data_root,
        label_to_index=label_to_index,
        split_sizes={key: len(loader.dataset) for key, loader in dataloaders.items()},
        window_size=window_size,
        step_size=step_size or window_size,
        samples_per_label=per_label,
    )
    return dataloaders, summary


def scan_data_root_v1(root: Path) -> None:
    """Utility helper that mimics the user's sample directory listing output."""

    files = discover_npz_files_v1(root)
    print(f"Scanning directory: '{root}'...")
    print()
    print(f"Found {len(files)} files. Listing all paths:")
    print("-" * 60)
    for path in files:
        print(path)
    print("-" * 60)
    print("Scan complete.")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect CWRU bearing data (v1)")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("CWRU_DATA_ROOT"),
        help="Root directory containing the downloaded CWRU `.npz` files.",
    )
    parser.add_argument("--scan", action="store_true", help="Print every discovered file path")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--test-split", type=float, default=0.0)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument(
        "--normalization",
        choices=["0-1", "-1-1", "mean-std"],
        default="-1-1",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    if args.data_root is None:
        raise ValueError("`--data-root` must be provided or set via CWRU_DATA_ROOT")
    data_root = Path(args.data_root)

    if args.scan:
        scan_data_root_v1(data_root)
        return

    dataloaders, summary = build_dataloaders_v1(
        data_root=data_root,
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


if __name__ == "__main__":  # pragma: no cover - manual use only
    main()
