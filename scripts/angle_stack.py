from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iceberg.data import load_or_build_cache  # noqa: E402
from iceberg.metrics import binary_log_loss  # noqa: E402


@dataclass
class PredictionInputs:
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    y: np.ndarray
    train_preds: np.ndarray
    test_preds: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Layer-2 stacking with incidence-angle group features."
    )
    parser.add_argument("--oof", nargs="+", type=Path, required=True, help="Full train OOF prediction CSVs.")
    parser.add_argument("--sub", nargs="+", type=Path, required=True, help="Matching test submission CSVs.")
    parser.add_argument("--model", choices=["lgbm", "logreg"], default="lgbm")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--real-decimals-max", type=int, default=4)
    parser.add_argument("--no-test-real-filter", action="store_true")
    parser.add_argument("--neighbors", nargs="+", type=int, default=[3, 5, 11, 23, 51])
    parser.add_argument("--radii", nargs="+", type=float, default=[0.005, 0.01, 0.025, 0.05, 0.1])
    parser.add_argument("--clip-grid", nargs="+", type=float, default=[0.001, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--fixed-clip", type=float, default=0.0, help="Use this low/high clip instead of CV-selected clip.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "artifacts" / "reports")
    return parser.parse_args()


def read_prediction_inputs(oof_paths: list[Path], sub_paths: list[Path]) -> PredictionInputs:
    if len(oof_paths) != len(sub_paths):
        raise ValueError("--oof and --sub must have the same number of files")

    train_arrays, test_arrays = load_or_build_cache(ROOT / "data" / "processed", ROOT / "data" / "cache", include_test=True)
    train_frame = pd.DataFrame(
        {
            "id": train_arrays.ids.astype(str),
            "is_iceberg": train_arrays.y.astype(int),
            "angle": train_arrays.angles.astype(float),
            "angle_decimals": train_arrays.angle_decimals.astype(int),
        }
    )
    test_frame = pd.DataFrame(
        {
            "id": test_arrays.ids.astype(str),
            "angle": test_arrays.angles.astype(float),
            "angle_decimals": test_arrays.angle_decimals.astype(int),
        }
    )

    for idx, path in enumerate(oof_paths):
        frame = pd.read_csv(path)[["id", "is_iceberg", "prediction"]].rename(columns={"prediction": f"pred_{idx}"})
        train_frame = train_frame.merge(frame, on=["id", "is_iceberg"], how="left")
    for idx, path in enumerate(sub_paths):
        frame = pd.read_csv(path)[["id", "is_iceberg"]].rename(columns={"is_iceberg": f"pred_{idx}"})
        test_frame = test_frame.merge(frame, on="id", how="left")

    pred_cols = [f"pred_{idx}" for idx in range(len(oof_paths))]
    missing_train = int(train_frame[pred_cols].isna().any(axis=1).sum())
    missing_test = int(test_frame[pred_cols].isna().any(axis=1).sum())
    if missing_train or missing_test:
        raise ValueError(
            "Angle stacking requires full OOF/submission coverage. "
            f"Missing rows: train={missing_train}, test={missing_test}."
        )

    y = train_frame["is_iceberg"].to_numpy(dtype=np.int64)
    train_preds = train_frame[pred_cols].to_numpy(dtype=np.float64)
    test_preds = test_frame[pred_cols].to_numpy(dtype=np.float64)
    train_preds = np.clip(train_preds, 1e-6, 1.0 - 1e-6)
    test_preds = np.clip(test_preds, 1e-6, 1.0 - 1e-6)
    return PredictionInputs(train_frame, test_frame, y, train_preds, test_preds)


def prediction_stats(preds: np.ndarray) -> tuple[np.ndarray, list[str]]:
    logits = np.log(preds / (1.0 - preds))
    stats = [
        ("pred_mean", preds.mean(axis=1)),
        ("pred_median", np.median(preds, axis=1)),
        ("pred_min", preds.min(axis=1)),
        ("pred_max", preds.max(axis=1)),
        ("pred_std", preds.std(axis=1)),
        ("pred_range", preds.max(axis=1) - preds.min(axis=1)),
        ("logit_mean", logits.mean(axis=1)),
        ("logit_std", logits.std(axis=1)),
    ]
    values = [preds[:, idx] for idx in range(preds.shape[1])]
    names = [f"pred_{idx}" for idx in range(preds.shape[1])]
    values.extend(arr for _, arr in stats)
    names.extend(name for name, _ in stats)
    return np.vstack(values).T.astype(np.float64), names


def filled_angles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    missing = np.isnan(values)
    filled = np.where(missing, 0.0, values).astype(np.float64)
    return filled, missing.astype(np.float64)


def group_stats_for_key(
    keys: np.ndarray,
    values: np.ndarray,
    base_values: np.ndarray,
    y_all: np.ndarray,
    known_label_mask: np.ndarray,
    participant_mask: np.ndarray,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    groups: dict[object, np.ndarray] = {}
    participant_indices = np.flatnonzero(participant_mask)
    for key in np.unique(keys[participant_mask]):
        groups[key] = participant_indices[keys[participant_indices] == key]

    n_rows = len(keys)
    out = np.zeros((n_rows, 9), dtype=np.float64)
    for row_idx, key in enumerate(keys):
        idxs = groups.get(key)
        if idxs is None or len(idxs) == 0:
            out[row_idx, :5] = [base_values[row_idx], base_values[row_idx], base_values[row_idx], base_values[row_idx], 0.0]
            out[row_idx, 5:] = [0.0, 0.0, 0.5, 0.5]
            continue

        row_values = values[idxs].copy()
        own_positions = np.flatnonzero(idxs == row_idx)
        if len(own_positions) and known_label_mask[row_idx]:
            row_values[own_positions[0]] = base_values[row_idx]

        label_idxs = idxs[known_label_mask[idxs]]
        if known_label_mask[row_idx]:
            label_idxs = label_idxs[label_idxs != row_idx]
        label_count = len(label_idxs)
        if label_count:
            label_mean = float(y_all[label_idxs].mean())
            label_smooth = float((y_all[label_idxs].sum() + 1.0) / (label_count + 2.0))
        else:
            label_mean = 0.5
            label_smooth = 0.5

        out[row_idx] = [
            float(row_values.mean()),
            float(np.median(row_values)),
            float(row_values.min()),
            float(row_values.max()),
            float(row_values.std()),
            float(len(idxs)),
            float(label_count),
            label_mean,
            label_smooth,
        ]

    names = [
        f"{prefix}_mean",
        f"{prefix}_median",
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_std",
        f"{prefix}_count",
        f"{prefix}_known_label_count",
        f"{prefix}_known_label_mean",
        f"{prefix}_known_label_smooth",
    ]
    return out, names


def neighbor_features(
    angles: np.ndarray,
    values: np.ndarray,
    participant_mask: np.ndarray,
    neighbors: Iterable[int],
) -> tuple[np.ndarray, list[str]]:
    participant_indices = np.flatnonzero(participant_mask)
    participant_angles = angles[participant_indices].reshape(-1, 1)
    max_neighbors = min(max(neighbors) + 1, len(participant_indices))
    model = NearestNeighbors(n_neighbors=max_neighbors, algorithm="brute")
    model.fit(participant_angles)
    distances, positions = model.kneighbors(angles.reshape(-1, 1), return_distance=True)

    outputs: list[np.ndarray] = []
    names: list[str] = []
    for k in neighbors:
        mean_values = np.zeros(len(angles), dtype=np.float64)
        weighted_values = np.zeros(len(angles), dtype=np.float64)
        for row_idx in range(len(angles)):
            idxs = participant_indices[positions[row_idx]]
            dists = distances[row_idx]
            keep = idxs != row_idx
            idxs = idxs[keep][:k]
            dists = dists[keep][:k]
            if len(idxs) == 0:
                mean_values[row_idx] = 0.5
                weighted_values[row_idx] = 0.5
                continue
            mean_values[row_idx] = float(values[idxs].mean())
            weights = 1.0 / (dists + 1e-6)
            weighted_values[row_idx] = float(np.sum(weights * values[idxs]) / np.sum(weights))
        outputs.extend([mean_values, weighted_values])
        names.extend([f"angle_knn{k}_mean", f"angle_knn{k}_weighted"])
    return np.vstack(outputs).T, names


def radius_features(
    angles: np.ndarray,
    values: np.ndarray,
    participant_mask: np.ndarray,
    radii: Iterable[float],
) -> tuple[np.ndarray, list[str]]:
    participant_indices = np.flatnonzero(participant_mask)
    participant_angles = angles[participant_indices]
    outputs: list[np.ndarray] = []
    names: list[str] = []
    for radius in radii:
        means = np.zeros(len(angles), dtype=np.float64)
        counts = np.zeros(len(angles), dtype=np.float64)
        for row_idx, angle in enumerate(angles):
            idxs = participant_indices[np.abs(participant_angles - angle) <= radius]
            idxs = idxs[idxs != row_idx]
            counts[row_idx] = len(idxs)
            means[row_idx] = 0.5 if len(idxs) == 0 else float(values[idxs].mean())
        outputs.extend([means, np.log1p(counts)])
        names.extend([f"angle_radius{radius:g}_mean", f"angle_radius{radius:g}_log_count"])
    return np.vstack(outputs).T, names


def build_angle_features(
    inputs: PredictionInputs,
    known_label_indices: np.ndarray,
    real_decimals_max: int,
    filter_test_real: bool,
    neighbors: list[int],
    radii: list[float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n_train = len(inputs.train_frame)
    y_all = np.concatenate([inputs.y.astype(np.float64), np.zeros(len(inputs.test_frame), dtype=np.float64)])
    all_preds = np.vstack([inputs.train_preds, inputs.test_preds])
    pred_features, pred_names = prediction_stats(all_preds)
    base_values = pred_features[:, pred_names.index("pred_mean")]

    train_angles, train_missing = filled_angles(inputs.train_frame["angle"].to_numpy(dtype=np.float64))
    test_angles, test_missing = filled_angles(inputs.test_frame["angle"].to_numpy(dtype=np.float64))
    all_angles = np.concatenate([train_angles, test_angles])
    all_missing = np.concatenate([train_missing, test_missing])
    all_decimals = np.concatenate(
        [
            inputs.train_frame["angle_decimals"].to_numpy(dtype=np.float64),
            inputs.test_frame["angle_decimals"].to_numpy(dtype=np.float64),
        ]
    )

    known_label_mask = np.zeros(len(all_angles), dtype=bool)
    known_label_mask[known_label_indices] = True
    values = base_values.copy()
    values[known_label_indices] = inputs.y[known_label_indices]

    test_real = inputs.test_frame["angle_decimals"].to_numpy(dtype=int) <= real_decimals_max
    if not filter_test_real:
        test_real = np.ones(len(inputs.test_frame), dtype=bool)
    participant_mask = np.concatenate([np.ones(n_train, dtype=bool), test_real])

    feature_blocks = [pred_features]
    feature_names = list(pred_names)
    angle_mean = float(train_angles.mean())
    angle_std = float(train_angles.std())
    if angle_std < 1e-6:
        angle_std = 1.0
    angle_block = np.vstack([(all_angles - angle_mean) / angle_std, all_missing, all_decimals]).T
    feature_blocks.append(angle_block)
    feature_names.extend(["inc_angle_scaled", "inc_angle_missing", "inc_angle_decimals"])

    for keys, prefix in [
        (all_angles, "angle_exact"),
        (np.round(all_angles, 4), "angle_round4"),
    ]:
        block, names = group_stats_for_key(
            keys=keys,
            values=values,
            base_values=base_values,
            y_all=y_all,
            known_label_mask=known_label_mask,
            participant_mask=participant_mask,
            prefix=prefix,
        )
        feature_blocks.append(block)
        feature_names.extend(names)

    block, names = neighbor_features(all_angles, values, participant_mask, neighbors)
    feature_blocks.append(block)
    feature_names.extend(names)
    block, names = radius_features(all_angles, values, participant_mask, radii)
    feature_blocks.append(block)
    feature_names.extend(names)

    features = np.concatenate(feature_blocks, axis=1)
    features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float64)
    return features[:n_train], features[n_train:], feature_names


def make_estimator(model_name: str, seed: int):
    if model_name == "lgbm":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                objective="binary",
                max_depth=3,
                n_estimators=70,
                learning_rate=0.1,
                min_child_samples=40,
                subsample=0.95,
                colsample_bytree=0.7,
                random_state=seed,
                verbose=-1,
            )
        except Exception as exc:
            raise RuntimeError("LightGBM is unavailable; install lightgbm or use --model logreg.") from exc
    if model_name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=5000, random_state=seed))
    raise ValueError(f"Unknown model: {model_name}")


def cross_val_predict_layer2(
    inputs: PredictionInputs,
    args: argparse.Namespace,
) -> np.ndarray:
    oof = np.zeros(len(inputs.y), dtype=np.float64)
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold_idx, (train_idx, valid_idx) in enumerate(folds.split(np.zeros(len(inputs.y)), inputs.y)):
        x_train_all, _, _ = build_angle_features(
            inputs,
            known_label_indices=train_idx,
            real_decimals_max=args.real_decimals_max,
            filter_test_real=not args.no_test_real_filter,
            neighbors=args.neighbors,
            radii=args.radii,
        )
        estimator = clone(make_estimator(args.model, args.seed + fold_idx))
        estimator.fit(x_train_all[train_idx], inputs.y[train_idx])
        oof[valid_idx] = estimator.predict_proba(x_train_all[valid_idx])[:, 1]
        fold_loss = binary_log_loss(inputs.y[valid_idx], oof[valid_idx])
        print(f"strict fold {fold_idx}: log_loss={fold_loss:.6f}")
    return oof


def clip_table(y: np.ndarray, pred: np.ndarray, lows: Iterable[float]) -> list[dict[str, float]]:
    rows = []
    for low in lows:
        low = float(low)
        high = 1.0 - low
        rows.append({"low": low, "high": high, "log_loss": binary_log_loss(y, np.clip(pred, low, high))})
    return rows


def select_clip(strict_table: list[dict[str, float]], fixed_clip: float) -> tuple[float, float]:
    if fixed_clip > 0:
        return fixed_clip, 1.0 - fixed_clip
    best = min(strict_table, key=lambda row: row["log_loss"])
    return float(best["low"]), float(best["high"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    inputs = read_prediction_inputs(args.oof, args.sub)
    test_real_like = int((inputs.test_frame["angle_decimals"].to_numpy(dtype=int) <= args.real_decimals_max).sum())
    test_synthetic_like = int(len(inputs.test_frame) - test_real_like)
    print(
        "loaded predictions: "
        f"train={len(inputs.train_frame)} test={len(inputs.test_frame)} models={inputs.train_preds.shape[1]}"
    )
    print(f"test real-like rows: {test_real_like}; synthetic-like rows: {test_synthetic_like}")

    strict_oof = cross_val_predict_layer2(inputs, args)

    x_train_full, x_test_full, feature_names = build_angle_features(
        inputs,
        known_label_indices=np.arange(len(inputs.y)),
        real_decimals_max=args.real_decimals_max,
        filter_test_real=not args.no_test_real_filter,
        neighbors=args.neighbors,
        radii=args.radii,
    )
    final_estimator = make_estimator(args.model, args.seed)
    final_estimator.fit(x_train_full, inputs.y)
    test_pred = final_estimator.predict_proba(x_test_full)[:, 1]

    strict_clips = clip_table(inputs.y, strict_oof, args.clip_grid)
    clip_low, clip_high = select_clip(strict_clips, args.fixed_clip)

    strict_loss = binary_log_loss(inputs.y, np.clip(strict_oof, clip_low, clip_high))
    test_pred_clipped = np.clip(test_pred, clip_low, clip_high)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_label = args.model
    prefix = f"angle_stack_{model_label}_{len(args.oof)}models_{stamp}"

    pd.DataFrame(
        {"id": inputs.train_frame["id"], "is_iceberg": inputs.y, "prediction": np.clip(strict_oof, clip_low, clip_high)}
    ).to_csv(args.output_dir / f"oof_strict_{prefix}.csv", index=False)
    pd.DataFrame({"id": inputs.test_frame["id"], "is_iceberg": test_pred_clipped}).to_csv(
        args.output_dir / f"submission_{prefix}.csv", index=False
    )

    metrics = {
        "strict_cv_log_loss": strict_loss,
        "selected_clip": {"low": clip_low, "high": clip_high},
        "strict_clip_table": strict_clips,
        "model": model_label,
        "folds": args.folds,
        "seed": args.seed,
        "oof_files": [str(path) for path in args.oof],
        "submission_files": [str(path) for path in args.sub],
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "test_real_like_rows": test_real_like,
        "test_synthetic_like_rows": test_synthetic_like,
        "test_real_filter": not args.no_test_real_filter,
        "neighbors": args.neighbors,
        "radii": args.radii,
    }
    with (args.report_dir / f"metrics_{prefix}.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"strict CV log_loss: {strict_loss:.6f}")
    print(f"selected clip: [{clip_low}, {clip_high}]")
    print(f"saved outputs with prefix: {prefix}")


if __name__ == "__main__":
    main()
