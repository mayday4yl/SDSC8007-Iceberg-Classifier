from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import re
from scipy.ndimage import median_filter
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


IMAGE_SIZE = 75


@dataclass
class IcebergArrays:
    ids: np.ndarray
    images: np.ndarray
    angles: np.ndarray
    angle_decimals: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None


def _read_angles(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").astype("float32").to_numpy()


def _scan_angle_decimals(path: Path) -> np.ndarray:
    """Read inc_angle decimal precision directly from JSON text.

    Pandas parses `inc_angle` as floats, which is fine for modeling but loses the
    raw-token precision signal used to separate real and machine-generated test
    rows in this competition.
    """
    pattern = re.compile(rb'"inc_angle"\s*:\s*("[^"]+"|[^,}\]]+)')
    decimals: list[int] = []
    with path.open("rb") as handle:
        carry = b""
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            data = carry + chunk
            scan = data[:-128]
            for match in pattern.finditer(scan):
                token = match.group(1).strip().strip(b'"')
                if token.lower() == b"na":
                    decimals.append(-1)
                elif b"." in token:
                    decimals.append(len(token.split(b".", 1)[1]))
                else:
                    decimals.append(0)
            carry = data[-128:]
        for match in pattern.finditer(carry):
            token = match.group(1).strip().strip(b'"')
            if token.lower() == b"na":
                decimals.append(-1)
            elif b"." in token:
                decimals.append(len(token.split(b".", 1)[1]))
            else:
                decimals.append(0)
    return np.asarray(decimals, dtype=np.int16)


def _read_bands(frame: pd.DataFrame) -> np.ndarray:
    band_1 = np.stack(
        frame["band_1"].map(lambda x: np.asarray(x, dtype=np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE))
    )
    band_2 = np.stack(
        frame["band_2"].map(lambda x: np.asarray(x, dtype=np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE))
    )
    return np.stack([band_1, band_2], axis=1).astype("float32", copy=False)


def load_json_arrays(path: Path, is_train: bool) -> IcebergArrays:
    frame = pd.read_json(path)
    ids = frame["id"].astype(str).to_numpy()
    images = _read_bands(frame)
    angles = _read_angles(frame["inc_angle"])
    angle_decimals = _scan_angle_decimals(path)
    if len(angle_decimals) != len(frame):
        raise ValueError(f"Parsed {len(angle_decimals)} inc_angle tokens for {len(frame)} rows in {path}")
    y = frame["is_iceberg"].astype("float32").to_numpy() if is_train else None
    return IcebergArrays(ids=ids, images=images, angles=angles, angle_decimals=angle_decimals, y=y)


def save_arrays(path: Path, arrays: IcebergArrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ids": arrays.ids.astype("U32"), "images": arrays.images, "angles": arrays.angles}
    if arrays.angle_decimals is not None:
        payload["angle_decimals"] = arrays.angle_decimals.astype(np.int16)
    if arrays.y is not None:
        payload["y"] = arrays.y
    np.savez_compressed(path, **payload)


def load_arrays(path: Path) -> IcebergArrays:
    data = np.load(path, allow_pickle=False)
    y = data["y"] if "y" in data.files else None
    angle_decimals = data["angle_decimals"] if "angle_decimals" in data.files else None
    return IcebergArrays(ids=data["ids"], images=data["images"], angles=data["angles"], angle_decimals=angle_decimals, y=y)


def load_or_build_cache(
    data_dir: Path,
    cache_dir: Path,
    include_test: bool = True,
) -> tuple[IcebergArrays, Optional[IcebergArrays]]:
    train_cache = cache_dir / "train_arrays.npz"
    test_cache = cache_dir / "test_arrays.npz"

    if train_cache.exists():
        try:
            train = load_arrays(train_cache)
            if train.angle_decimals is None:
                raise ValueError("train cache missing angle_decimals")
        except ValueError:
            train = load_json_arrays(data_dir / "train.json", is_train=True)
            save_arrays(train_cache, train)
    else:
        train = load_json_arrays(data_dir / "train.json", is_train=True)
        save_arrays(train_cache, train)

    test = None
    if include_test:
        if test_cache.exists():
            try:
                test = load_arrays(test_cache)
                if test.angle_decimals is None:
                    raise ValueError("test cache missing angle_decimals")
            except ValueError:
                test = load_json_arrays(data_dir / "test.json", is_train=False)
                save_arrays(test_cache, test)
        else:
            test = load_json_arrays(data_dir / "test.json", is_train=False)
            save_arrays(test_cache, test)

    return train, test


def make_image_channels(raw_images: np.ndarray, mode: str = "linear") -> np.ndarray:
    """Create model channels from raw Kaggle dual-band images.

    Supported modes:
    - linear: exp(x / 20), then b1/b2/avg/diff.
    - db: raw dB values, then b1/b2/avg/diff.
    - linear3: exp(x / 20), then b1/b2/b1+b2.
    - db3: raw dB values, then b1/b2/b1+b2.
    - linear_median: linear channels after 3x3 median filtering.
    - db_median: raw dB channels after 3x3 median filtering.
    """
    denoise = mode.endswith("_median")
    base_mode = mode.removesuffix("_median")

    if base_mode in {"linear", "linear3"}:
        base = np.exp(raw_images / 20.0)
    elif base_mode in {"db", "db3"}:
        base = raw_images.astype("float32", copy=False)
    else:
        raise ValueError(f"Unknown image mode: {mode}")
    if denoise:
        base = median_filter(base, size=(1, 1, 3, 3), mode="nearest")

    band_1 = base[:, 0]
    band_2 = base[:, 1]
    if base_mode.endswith("3"):
        combined = band_1 + band_2
        channels = np.stack([band_1, band_2, combined], axis=1)
    else:
        avg = (band_1 + band_2) / 2.0
        diff = band_1 - band_2
        channels = np.stack([band_1, band_2, avg, diff], axis=1)
    return channels.astype("float32", copy=False)


def standardize_images(
    train_images: np.ndarray,
    test_images: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, list[float]]]:
    means = train_images.mean(axis=(0, 2, 3), keepdims=True)
    stds = train_images.std(axis=(0, 2, 3), keepdims=True)
    stds = np.where(stds < 1e-6, 1.0, stds)

    train_scaled = (train_images - means) / stds
    test_scaled = None if test_images is None else (test_images - means) / stds
    scaler = {
        "mean": means.reshape(-1).astype(float).tolist(),
        "std": stds.reshape(-1).astype(float).tolist(),
    }
    return train_scaled.astype("float32"), None if test_scaled is None else test_scaled.astype("float32"), scaler


def preprocess_images(
    train_raw: np.ndarray,
    test_raw: Optional[np.ndarray] = None,
    mode: str = "linear",
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, list[float]]]:
    train_images = make_image_channels(train_raw, mode=mode)
    test_images = None if test_raw is None else make_image_channels(test_raw, mode=mode)
    return standardize_images(train_images, test_images)


def prepare_angles(
    train_angles: np.ndarray,
    test_angles: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, float]]:
    train_missing = np.isnan(train_angles).astype("float32")
    median = float(np.nanmedian(train_angles))
    filled_train = np.where(np.isnan(train_angles), median, train_angles).astype("float32")
    mean = float(filled_train.mean())
    std = float(filled_train.std())
    if std < 1e-6:
        std = 1.0

    train_features = np.stack([(filled_train - mean) / std, train_missing], axis=1).astype("float32")

    test_features = None
    if test_angles is not None:
        test_missing = np.isnan(test_angles).astype("float32")
        filled_test = np.where(np.isnan(test_angles), median, test_angles).astype("float32")
        test_features = np.stack([(filled_test - mean) / std, test_missing], axis=1).astype("float32")

    return train_features, test_features, {"median": median, "mean": mean, "std": std}


def make_angle_groups(angles: np.ndarray, decimals: int = 4) -> np.ndarray:
    filled = np.where(np.isnan(angles), -1.0, angles).astype("float32")
    return np.round(filled, decimals=decimals).astype(str)


def make_folds(
    y: np.ndarray,
    n_splits: int,
    seed: int,
    strategy: str = "stratified",
    angles: Optional[np.ndarray] = None,
    angle_group_decimals: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(y))
    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(indices, y.astype(int)))
    if strategy == "angle_group":
        if angles is None:
            raise ValueError("angle_group fold strategy requires angles")
        groups = make_angle_groups(angles, decimals=angle_group_decimals)
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(indices, y.astype(int), groups=groups))
    raise ValueError(f"Unknown fold strategy: {strategy}")


def image_stat_features(raw_images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    db = raw_images.astype("float32", copy=False)
    linear = np.exp(db / 20.0)
    channel_map = {
        "db_b1": db[:, 0],
        "db_b2": db[:, 1],
        "db_avg": (db[:, 0] + db[:, 1]) / 2.0,
        "db_diff": db[:, 0] - db[:, 1],
        "lin_b1": linear[:, 0],
        "lin_b2": linear[:, 1],
        "lin_avg": (linear[:, 0] + linear[:, 1]) / 2.0,
        "lin_diff": linear[:, 0] - linear[:, 1],
    }

    values: list[np.ndarray] = []
    names: list[str] = []
    percentiles = [1, 5, 25, 50, 75, 95, 99]

    for name, image in channel_map.items():
        if name in {"polar_ratio", "polar_contrast"}:
            image = np.log1p(np.clip(image, 0.0, 1e6))
        flat = image.reshape(image.shape[0], -1)
        stats = [
            ("mean", flat.mean(axis=1)),
            ("std", flat.std(axis=1)),
            ("min", flat.min(axis=1)),
            ("max", flat.max(axis=1)),
        ]
        for pct in percentiles:
            stats.append((f"p{pct}", np.percentile(flat, pct, axis=1)))
        for suffix, arr in stats:
            values.append(arr.astype("float32"))
            names.append(f"{name}_{suffix}")

    features = np.vstack(values).T
    return np.nan_to_num(features, copy=False).astype("float32"), names


def physical_features(raw_images: np.ndarray, angles: Optional[np.ndarray] = None) -> tuple[np.ndarray, list[str]]:
    db = raw_images.astype("float32", copy=False)
    linear = np.exp(db / 20.0)
    b1 = linear[:, 0]
    b2 = linear[:, 1]
    eps = 1e-6
    ratio = b1 / (b2 + eps)
    contrast = np.maximum(b1, b2) / (np.minimum(b1, b2) + eps)
    total = b1 + b2
    diff = b1 - b2

    channel_map = {
        "polar_ratio": ratio,
        "polar_contrast": contrast,
        "linear_total": total,
        "linear_diff": diff,
    }
    values: list[np.ndarray] = []
    names: list[str] = []
    percentiles = [50, 90, 95, 99]
    for name, image in channel_map.items():
        flat = image.reshape(image.shape[0], -1)
        stats = [
            ("mean", flat.mean(axis=1)),
            ("std", flat.std(axis=1)),
            ("max", flat.max(axis=1)),
            ("max_mean_ratio", flat.max(axis=1) / (flat.mean(axis=1) + eps)),
        ]
        for pct in percentiles:
            stats.append((f"p{pct}", np.percentile(flat, pct, axis=1)))
        for suffix, arr in stats:
            values.append(arr.astype("float32"))
            names.append(f"{name}_{suffix}")

    if angles is not None:
        missing = np.isnan(angles).astype("float32")
        median = float(np.nanmedian(angles))
        filled = np.where(np.isnan(angles), median, angles).astype("float32")
        values.extend([filled, missing])
        names.extend(["inc_angle", "inc_angle_missing"])

    features = np.vstack(values).T
    return np.nan_to_num(features, copy=False, posinf=1e6, neginf=-1e6).astype("float32"), names


def engineered_features(raw_images: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, list[str]]:
    image_features, image_names = image_stat_features(raw_images)
    values = [image_features[:, idx] for idx in range(image_features.shape[1])]
    names = list(image_names)

    missing = np.isnan(angles).astype("float32")
    median = float(np.nanmedian(angles))
    filled = np.where(np.isnan(angles), median, angles).astype("float32")
    values.extend([filled, missing])
    names.extend(["inc_angle", "inc_angle_missing"])

    features = np.vstack(values).T
    return np.nan_to_num(features, copy=False).astype("float32"), names
