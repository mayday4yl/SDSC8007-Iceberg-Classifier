from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iceberg.metrics import binary_log_loss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify final OOF log loss from saved prediction artifacts.")
    parser.add_argument(
        "--oof",
        type=Path,
        default=ROOT / "artifacts" / "predictions" / "oof_blend_17models_20260503_164351.csv",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT / "artifacts" / "reports" / "metrics_blend_17models_20260503_164351.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oof = pd.read_csv(args.oof)
    cv = binary_log_loss(oof["is_iceberg"].to_numpy(), oof["prediction"].to_numpy())
    print(f"recomputed_cv_log_loss: {cv:.15f}")

    if args.metrics.exists():
        metrics = json.load(args.metrics.open())
        recorded = metrics.get("cv_log_loss")
        print(f"recorded_cv_log_loss:   {recorded:.15f}")
        print(f"absolute_difference:    {abs(cv - recorded):.15g}")


if __name__ == "__main__":
    main()
