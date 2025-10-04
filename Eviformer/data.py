import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from SequenceDatasets import dataset
from sequence_aug import *
from tqdm import tqdm
from torch.utils.data import DataLoader

signal_size = 1024

NPZ_DEFAULT_RPM = "1797"
NPZ_DEFAULT_CHANNEL = "DE12"
NPZ_DEFAULT_RPM_DIR = f"{NPZ_DEFAULT_RPM} RPM"
NPZ_NORMAL_TEMPLATE = "{rpm}_Normal.npz"
_NPZ_FAULT_SPECS: Sequence[Tuple[str, str]] = (
    ("B", "7"),
    ("B", "14"),
    ("B", "21"),
    ("IR", "7"),
    ("IR", "14"),
    ("IR", "21"),
    ("OR@6", "7"),
    ("OR@6", "14"),
    ("OR@6", "21"),
)


def _build_npz_filename(rpm: str, fault_prefix: str, defect_size: str, channel: str) -> str:
    return f"{rpm}_{fault_prefix}_{defect_size}_{channel}.npz"


def _load_npz_signal(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    with np.load(path) as archive:
        for key in ("signal", "data", "DE_time", "FE_time", "BA_time"):
            if key in archive:
                return np.asarray(archive[key])
        return np.asarray(archive[archive.files[0]])


def _window_signal(signal: np.ndarray, label: int) -> Tuple[List[np.ndarray], List[int]]:
    signal = np.asarray(signal)
    data: List[np.ndarray] = []
    labels: List[int] = []

    if signal.ndim == 1:
        start, end = 0, signal_size
        while end <= signal.shape[0]:
            data.append(signal[start:end].astype(np.float32))
            labels.append(label)
            start += signal_size
            end += signal_size
    elif signal.ndim == 2:
        for row in signal:
            row = np.asarray(row).ravel()
            if row.size < signal_size:
                continue
            data.append(row[:signal_size].astype(np.float32))
            labels.append(label)
    else:
        raise ValueError(f"Unsupported signal shape {signal.shape}")

    return data, labels


def data_load_npz(filename: str, label: int) -> Tuple[List[np.ndarray], List[int]]:
    signal = _load_npz_signal(filename)
    return _window_signal(signal, label)


def get_files(
    root: str,
    test: bool = False,
    rpm_dir: str = NPZ_DEFAULT_RPM_DIR,
    channel: str = NPZ_DEFAULT_CHANNEL,
) -> List[List[np.ndarray]]:
    data_dir = os.path.join(root, rpm_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Unable to locate data directory: {data_dir}")

    rpm_value = rpm_dir.split()[0]
    normal_path = os.path.join(data_dir, NPZ_NORMAL_TEMPLATE.format(rpm=rpm_value))
    data, lab = data_load_npz(normal_path, label=0)

    for idx, (fault_prefix, defect_size) in enumerate(tqdm(_NPZ_FAULT_SPECS)):
        fault_filename = _build_npz_filename(rpm_value, fault_prefix, defect_size, channel)
        fault_path = os.path.join(data_dir, fault_filename)
        fault_data, fault_labels = data_load_npz(fault_path, label=idx + 1)
        data.extend(fault_data)
        lab.extend(fault_labels)

    return [data, lab]


def data_transforms(dataset_type: str = "train", normlize_type: str = "-1-1"):
    transforms = {
        "train": Compose([Reshape(), Normalize(normlize_type), Retype()]),
        "val": Compose([Reshape(), Normalize(normlize_type), Retype()]),
    }
    return transforms[dataset_type]


def load_cwru_dataset(
    data_dir: str,
    normlizetype: str,
    test: bool = False,
    rpm_dir: str = NPZ_DEFAULT_RPM_DIR,
    channel: str = NPZ_DEFAULT_CHANNEL,
):
    list_data = get_files(data_dir, test=test, rpm_dir=rpm_dir, channel=channel)

    if test:
        return dataset(list_data=list_data, test=True, transform=None)

    data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
    train_pd, val_pd = train_test_split(
        data_pd,
        test_size=0.20,
        random_state=40,
        stratify=data_pd["label"],
    )
    train_dataset = dataset(list_data=train_pd, transform=data_transforms("train", normlizetype))
    val_dataset = dataset(list_data=val_pd, transform=data_transforms("val", normlizetype))
    return train_dataset, val_dataset


if __name__ == "__main__":
    base_dir = r"D:\\CWRU_Bearing_NumPy-main\\Data"
    data_train, data_val = load_cwru_dataset(base_dir, "0-1")

    dataloader_train = DataLoader(data_train, batch_size=16, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(data_val, batch_size=16, num_workers=0)

    dataloaders = {
        "train": dataloader_train,
        "val": dataloader_val,
    }

    digit_one, _ = data_val[5]
