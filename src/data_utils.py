from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import stft
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "healthy",
    "inner_race_fault",
    "outer_race_fault",
    "ball_fault",
]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def parse_fault_category(filename: str) -> str:
    name = Path(filename).name
    if name.startswith("Time_Normal"):
        return "healthy"
    if name.startswith("IR"):
        return "inner_race_fault"
    if name.startswith("OR"):
        return "outer_race_fault"
    if name.startswith("B"):
        return "ball_fault"
    raise ValueError(f"Unknown fault category for file: {filename}")


def _pick_drive_end_key(keys: list[str], stem: str) -> str:
    de_keys = [k for k in keys if k.endswith("_DE_time")]
    if not de_keys:
        raise ValueError(f"No drive-end channel found in MAT file stem={stem} keys={keys}")

    match = re.search(r"_(\d+)$", stem)
    if match:
        expected = f"X{match.group(1)}_DE_time"
        if expected in de_keys:
            return expected

    return sorted(de_keys)[0]


def load_drive_end_signal(mat_path: Path) -> np.ndarray:
    data = sio.loadmat(mat_path)
    keys = [k for k in data.keys() if not k.startswith("__")]
    selected_key = _pick_drive_end_key(keys, mat_path.stem)
    signal = np.asarray(data[selected_key], dtype=np.float32).squeeze()
    if signal.ndim != 1:
        signal = signal.reshape(-1)
    return signal


def sliding_windows(signal: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    if signal.shape[0] < window_size:
        return np.empty((0, window_size), dtype=np.float32)

    windows = [
        signal[start : start + window_size]
        for start in range(0, signal.shape[0] - window_size + 1, step_size)
    ]
    return np.asarray(windows, dtype=np.float32)


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mean) / std


def _resample_1d(x: np.ndarray, target_length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] == target_length:
        return x

    old_axis = np.linspace(0.0, 1.0, num=x.shape[0], dtype=np.float32)
    new_axis = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(new_axis, old_axis, x).astype(np.float32)


def fft_feature(window: np.ndarray, target_length: int) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(window))
    spectrum = np.log1p(spectrum)
    spectrum = _resample_1d(spectrum, target_length)
    return zscore_1d(spectrum)


def stft_feature(
    window: np.ndarray,
    fs: int,
    target_length: int,
    nperseg: int = 256,
    noverlap: int = 128,
) -> np.ndarray:
    _, _, zxx = stft(
        window,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    profile = np.abs(zxx).mean(axis=1)
    profile = np.log1p(profile)
    profile = _resample_1d(profile, target_length)
    return zscore_1d(profile)


def build_multichannel_sample(window: np.ndarray, fs: int, output_length: int) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    raw = zscore_1d(window)
    raw = _resample_1d(raw, output_length)

    fft_profile = fft_feature(window, output_length)
    stft_profile = stft_feature(window, fs=fs, target_length=output_length)

    return np.stack([raw, fft_profile, stft_profile], axis=0).astype(np.float32)


def build_dataset_from_raw(
    raw_dir: str | Path,
    window_size: int = 2048,
    step_size: int = 1024,
    fs: int = 48000,
    max_windows_per_file: int | None = None,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    raw_dir = Path(raw_dir)
    mat_files = sorted(raw_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {raw_dir}")

    rng = np.random.default_rng(random_seed)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    per_file_windows: dict[str, int] = {}

    for mat_file in mat_files:
        category = parse_fault_category(mat_file.name)
        label_idx = CLASS_TO_INDEX[category]

        signal = load_drive_end_signal(mat_file)
        windows = sliding_windows(signal, window_size=window_size, step_size=step_size)
        if windows.shape[0] == 0:
            continue

        if max_windows_per_file is not None and windows.shape[0] > max_windows_per_file:
            chosen = np.sort(
                rng.choice(windows.shape[0], size=max_windows_per_file, replace=False)
            )
            windows = windows[chosen]

        per_file_windows[mat_file.name] = int(windows.shape[0])

        for window in windows:
            features = build_multichannel_sample(window, fs=fs, output_length=window_size)
            X_list.append(features)
            y_list.append(label_idx)

    if not X_list:
        raise RuntimeError("No training samples were created from raw MAT files.")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    class_distribution = {
        CLASS_NAMES[idx]: int((y == idx).sum()) for idx in range(len(CLASS_NAMES))
    }

    metadata = {
        "num_samples": int(X.shape[0]),
        "input_shape": [int(v) for v in X.shape[1:]],
        "window_size": int(window_size),
        "step_size": int(step_size),
        "sampling_rate": int(fs),
        "raw_dir": str(raw_dir),
        "class_distribution": class_distribution,
        "per_file_windows": per_file_windows,
    }

    return X, y, CLASS_NAMES.copy(), metadata


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not (0.0 < test_size < 1.0 and 0.0 < val_size < 1.0):
        raise ValueError("test_size and val_size must be between 0 and 1.")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=adjusted_val_size,
        stratify=y_temp,
        random_state=random_seed,
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }


def summarize_feature_csv(csv_path: str | Path) -> dict[str, Any] | None:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    summary: dict[str, Any] = {
        "path": str(csv_path),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(col) for col in df.columns.tolist()],
    }

    if "fault" in df.columns:
        counts = df["fault"].value_counts().to_dict()
        summary["fault_counts"] = {str(k): int(v) for k, v in counts.items()}

    return summary
