from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iceberg.data import load_or_build_cache  # noqa: E402


def main() -> None:
    train, test = load_or_build_cache(ROOT / "data" / "processed", ROOT / "data" / "cache", include_test=True)
    print(f"train images: {train.images.shape}, labels: {train.y.shape}")
    print(f"test images: {test.images.shape}")
    print(f"train missing incidence angles: {int((train.angles != train.angles).sum())}")
    print(f"test missing incidence angles: {int((test.angles != test.angles).sum())}")
    if test.angle_decimals is not None:
        real_like = (test.angle_decimals >= 0) & (test.angle_decimals <= 4)
        synthetic_like = test.angle_decimals > 4
        print(f"test real-like inc_angle decimals <=4: {int(real_like.sum())}")
        print(f"test synthetic-like inc_angle decimals >4: {int(synthetic_like.sum())}")


if __name__ == "__main__":
    main()
