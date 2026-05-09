from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iceberg.metrics import binary_log_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend OOF and submission predictions.")
    parser.add_argument("--oof", nargs="+", type=Path, required=True)
    parser.add_argument("--sub", nargs="+", type=Path, required=True)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "artifacts" / "reports")
    return parser.parse_args()


def read_oof(paths: list[Path]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    merged = None
    for idx, path in enumerate(paths):
        frame = pd.read_csv(path)[["id", "is_iceberg", "prediction"]]
        frame = frame.rename(columns={"prediction": f"pred_{idx}"})
        merged = frame if merged is None else merged.merge(frame, on=["id", "is_iceberg"], how="inner")
    pred_cols = [f"pred_{idx}" for idx in range(len(paths))]
    merged = merged.dropna(subset=pred_cols)
    y = merged["is_iceberg"].to_numpy()
    preds = merged[pred_cols].to_numpy(dtype=np.float64)
    return merged, y, preds


def find_weights(y: np.ndarray, preds: np.ndarray, grid_step: float) -> np.ndarray:
    n_models = preds.shape[1]
    if n_models == 1:
        return np.ones(1, dtype=np.float64)
    if n_models == 2:
        grid = np.arange(0.0, 1.0 + 1e-12, grid_step)
        best_weight = 0.5
        best_loss = float("inf")
        for w in grid:
            blended = w * preds[:, 0] + (1.0 - w) * preds[:, 1]
            loss = binary_log_loss(y, blended)
            if loss < best_loss:
                best_loss = loss
                best_weight = float(w)
        return np.array([best_weight, 1.0 - best_weight], dtype=np.float64)

    def objective(weights: np.ndarray) -> float:
        blended = np.sum(preds * weights.reshape(1, -1), axis=1)
        return binary_log_loss(y, blended)

    initial = np.ones(n_models, dtype=np.float64) / n_models
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        return initial
    weights = np.clip(result.x, 0.0, 1.0)
    total = weights.sum()
    return initial if total <= 0 else weights / total


def read_submissions(paths: list[Path]) -> pd.DataFrame:
    merged = None
    for idx, path in enumerate(paths):
        frame = pd.read_csv(path)[["id", "is_iceberg"]].rename(columns={"is_iceberg": f"pred_{idx}"})
        merged = frame if merged is None else merged.merge(frame, on="id", how="inner")
    return merged


def main() -> None:
    args = parse_args()
    if len(args.oof) != len(args.sub):
        raise ValueError("--oof and --sub must have the same number of files")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    oof_frame, y, oof_preds = read_oof(args.oof)
    weights = find_weights(y, oof_preds, args.grid_step)
    blended_oof = np.clip(np.sum(oof_preds * weights.reshape(1, -1), axis=1), 1e-7, 1.0 - 1e-7)
    loss = binary_log_loss(y, blended_oof)

    sub_frame = read_submissions(args.sub)
    sub_cols = [f"pred_{idx}" for idx in range(len(args.sub))]
    blended_sub = np.clip(
        np.sum(sub_frame[sub_cols].to_numpy(dtype=np.float64) * weights.reshape(1, -1), axis=1),
        1e-5,
        1.0 - 1e-5,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"blend_{len(args.oof)}models_{stamp}"
    pd.DataFrame({"id": oof_frame["id"], "is_iceberg": y, "prediction": blended_oof}).to_csv(
        args.output_dir / f"oof_{prefix}.csv", index=False
    )
    pd.DataFrame({"id": sub_frame["id"], "is_iceberg": blended_sub}).to_csv(
        args.output_dir / f"submission_{prefix}.csv", index=False
    )
    metrics = {
        "cv_log_loss": loss,
        "weights": weights.tolist(),
        "oof_files": [str(path) for path in args.oof],
        "submission_files": [str(path) for path in args.sub],
        "n_oof_rows": int(len(oof_frame)),
    }
    with (args.report_dir / f"metrics_{prefix}.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"blend CV log_loss: {loss:.6f}")
    print(f"weights: {weights.tolist()}")
    print(f"saved outputs with prefix: {prefix}")


if __name__ == "__main__":
    main()
