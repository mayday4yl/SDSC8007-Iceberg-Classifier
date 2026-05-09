from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iceberg.data import image_stat_features, load_or_build_cache, make_folds, make_image_channels, standardize_images  # noqa: E402
from iceberg.metrics import binary_log_loss  # noqa: E402
from iceberg.models import build_model, count_parameters  # noqa: E402


class IcebergDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        angles: np.ndarray,
        y: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
        augment: bool = False,
    ) -> None:
        self.images = images
        self.angles = angles
        self.y = y
        self.sample_weights = sample_weights
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        if self.augment:
            image = augment_image(image)
        image_t = torch.from_numpy(np.ascontiguousarray(image)).float()
        angle_t = torch.from_numpy(self.angles[idx]).float()
        if self.y is None:
            return image_t, angle_t
        y_t = torch.tensor(self.y[idx], dtype=torch.float32)
        if self.sample_weights is None:
            return image_t, angle_t, y_t
        return image_t, angle_t, y_t, torch.tensor(self.sample_weights[idx], dtype=torch.float32)


def augment_image(image: np.ndarray) -> np.ndarray:
    out = image
    k = random.randint(0, 3)
    if k:
        out = np.rot90(out, k=k, axes=(-2, -1))
    if random.random() < 0.5:
        out = out[..., :, ::-1]
    if random.random() < 0.5:
        out = out[..., ::-1, :]
    if random.random() < 0.25:
        out = out + np.random.normal(0.0, 0.025, size=out.shape).astype("float32")
    return out.astype("float32", copy=False)


def d4_transform(x: torch.Tensor, index: int) -> torch.Tensor:
    if index == 0:
        return x
    if index == 1:
        return torch.rot90(x, 1, dims=(-2, -1))
    if index == 2:
        return torch.rot90(x, 2, dims=(-2, -1))
    if index == 3:
        return torch.rot90(x, 3, dims=(-2, -1))
    if index == 4:
        return torch.flip(x, dims=(-1,))
    if index == 5:
        return torch.flip(x, dims=(-2,))
    if index == 6:
        return torch.transpose(x, -2, -1)
    if index == 7:
        return torch.flip(torch.transpose(x, -2, -1), dims=(-1,))
    raise ValueError(index)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_angle_scaler(angles: np.ndarray) -> dict[str, float]:
    median = float(np.nanmedian(angles))
    filled = np.where(np.isnan(angles), median, angles).astype("float32")
    mean = float(filled.mean())
    std = float(filled.std())
    if std < 1e-6:
        std = 1.0
    return {"median": median, "mean": mean, "std": std}


def transform_angles(angles: np.ndarray, scaler: dict[str, float]) -> np.ndarray:
    missing = np.isnan(angles).astype("float32")
    filled = np.where(np.isnan(angles), scaler["median"], angles).astype("float32")
    scaled = (filled - scaler["mean"]) / scaler["std"]
    return np.stack([scaled, missing], axis=1).astype("float32")


def build_aux_features(
    train_images: np.ndarray,
    train_angles: np.ndarray,
    test_images: Optional[np.ndarray],
    test_angles: Optional[np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], dict[str, object]]:
    angle_scaler = fit_angle_scaler(train_angles[train_idx])
    angle_train = transform_angles(train_angles[train_idx], angle_scaler)
    angle_val = transform_angles(train_angles[val_idx], angle_scaler)
    angle_test = None if test_angles is None else transform_angles(test_angles, angle_scaler)

    scaler_info: dict[str, object] = {"angle_scaler": angle_scaler, "aux_mode": mode}
    if mode == "angle":
        return angle_train, angle_val, angle_test, scaler_info
    if mode != "stats":
        raise ValueError(f"Unknown aux feature mode: {mode}")

    stat_train_all, stat_names = image_stat_features(train_images)
    stat_test_all = None if test_images is None else image_stat_features(test_images)[0]
    stat_scaler = StandardScaler()
    stat_train = stat_scaler.fit_transform(stat_train_all[train_idx]).astype("float32")
    stat_val = stat_scaler.transform(stat_train_all[val_idx]).astype("float32")
    stat_test = None if stat_test_all is None else stat_scaler.transform(stat_test_all).astype("float32")

    aux_train = np.concatenate([angle_train, stat_train], axis=1).astype("float32")
    aux_val = np.concatenate([angle_val, stat_val], axis=1).astype("float32")
    aux_test = None if angle_test is None or stat_test is None else np.concatenate([angle_test, stat_test], axis=1).astype("float32")
    scaler_info.update(
        {
            "stat_feature_count": int(stat_train.shape[1]),
            "stat_feature_names": stat_names,
            "stat_mean": stat_scaler.mean_.astype(float).tolist(),
            "stat_scale": stat_scaler.scale_.astype(float).tolist(),
        }
    )
    return aux_train, aux_val, aux_test, scaler_info


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pseudo_labels(
    path: Optional[Path],
    test_ids: Optional[np.ndarray],
    angle_decimals: Optional[np.ndarray],
    low: float,
    high: float,
    max_samples: int,
    seed: int,
    real_decimals_max: int,
    filter_real: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if path is None or test_ids is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    frame = pd.read_csv(path)
    pred_map = dict(zip(frame["id"].astype(str), frame["is_iceberg"].astype(float)))
    preds = np.array([pred_map.get(str(sample_id), np.nan) for sample_id in test_ids], dtype=np.float32)
    mask = np.isfinite(preds) & ((preds <= low) | (preds >= high))
    if filter_real:
        if angle_decimals is None:
            raise ValueError("Pseudo-label real-test filtering requires angle_decimals in the test cache.")
        real_mask = (angle_decimals >= 0) & (angle_decimals <= real_decimals_max)
        mask &= real_mask
    idx = np.flatnonzero(mask)
    if max_samples > 0 and len(idx) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_samples, replace=False)
        idx.sort()
    labels = (preds[idx] >= high).astype("float32")
    return idx.astype(np.int64), labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal CNN with stratified CV.")
    parser.add_argument("--arch", choices=["resnet", "vgg", "film_resnet"], default="resnet")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable pretrained weights for architectures that support them.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-strategy", choices=["stratified", "angle_group"], default="stratified")
    parser.add_argument("--angle-group-decimals", type=int, default=4)
    parser.add_argument("--fold", type=int, default=-1, help="Train a single fold. Use -1 for all folds.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument(
        "--image-mode",
        choices=["linear", "db", "linear3", "db3", "linear_median", "db_median"],
        default="linear",
    )
    parser.add_argument(
        "--aux-features",
        choices=["angle", "stats"],
        default="stats",
        help="Auxiliary branch: incidence angle only, or angle plus image statistics.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tta", action="store_true", help="Average 8 D4 transforms for test predictions.")
    parser.add_argument("--val-tta", action="store_true", help="Use D4 TTA for validation scoring and OOF predictions.")
    parser.add_argument("--no-test", action="store_true", help="Skip test predictions to save time/memory.")
    parser.add_argument("--limit-train-samples", type=int, default=0, help="Small debugging subset before CV split.")
    parser.add_argument("--pseudo-sub", type=Path, default=None, help="Submission CSV used to create high-confidence pseudo-labels.")
    parser.add_argument("--pseudo-low", type=float, default=0.03)
    parser.add_argument("--pseudo-high", type=float, default=0.97)
    parser.add_argument("--pseudo-weight", type=float, default=0.35)
    parser.add_argument("--pseudo-max-samples", type=int, default=0, help="Limit pseudo samples for debugging. 0 keeps all selected.")
    parser.add_argument(
        "--pseudo-real-decimals-max",
        type=int,
        default=4,
        help="Keep pseudo-label candidates with raw inc_angle decimal precision <= this value.",
    )
    parser.add_argument(
        "--no-pseudo-real-filter",
        action="store_true",
        help="Disable inc_angle precision filtering for pseudo-label candidates.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "models")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "artifacts" / "reports")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    for batch in loader:
        if len(batch) == 4:
            images, angles, y, weights = batch
        else:
            images, angles, y = batch
            weights = torch.ones_like(y)
        images = images.to(device)
        angles = angles.to(device)
        y = y.to(device)
        weights = weights.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, angles)
        loss_values = loss_fn(logits, y)
        loss = (loss_values * weights).sum() / weights.sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * float(weights.sum().item())
        total_weight += float(weights.sum().item())
    return total_loss / max(total_weight, 1.0)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for batch in loader:
        if len(batch) == 4:
            images, angles, _, _ = batch
        elif len(batch) == 3:
            images, angles, _ = batch
        else:
            images, angles = batch
        images = images.to(device)
        angles = angles.to(device)
        if tta:
            probs = []
            for idx in range(8):
                logits = model(d4_transform(images, idx), angles)
                probs.append(torch.sigmoid(logits))
            pred = torch.stack(probs, dim=0).mean(dim=0)
        else:
            pred = torch.sigmoid(model(images, angles))
        outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs).astype("float32")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    include_test = (not args.no_test) or args.pseudo_sub is not None

    train, test = load_or_build_cache(ROOT / "data" / "processed", ROOT / "data" / "cache", include_test=include_test)

    if args.limit_train_samples > 0:
        rng = np.random.default_rng(args.seed)
        picks = rng.choice(np.arange(len(train.y)), size=min(args.limit_train_samples, len(train.y)), replace=False)
        train.images = train.images[picks]
        train.angles = train.angles[picks]
        train.y = train.y[picks]
        train.ids = train.ids[picks]

    train_channels = make_image_channels(train.images, mode=args.image_mode)
    test_channels = None if test is None else make_image_channels(test.images, mode=args.image_mode)
    pseudo_idx, pseudo_y = load_pseudo_labels(
        args.pseudo_sub,
        None if test is None else test.ids,
        None if test is None else test.angle_decimals,
        args.pseudo_low,
        args.pseudo_high,
        args.pseudo_max_samples,
        args.seed,
        args.pseudo_real_decimals_max,
        not args.no_pseudo_real_filter,
    )
    if args.pseudo_sub is not None:
        real_like = 0
        synthetic_like = 0
        if test is not None and test.angle_decimals is not None:
            real_like = int(((test.angle_decimals >= 0) & (test.angle_decimals <= args.pseudo_real_decimals_max)).sum())
            synthetic_like = int((test.angle_decimals > args.pseudo_real_decimals_max).sum())
        print(
            f"pseudo labels: {len(pseudo_idx)} selected from {args.pseudo_sub} "
            f"(low<={args.pseudo_low}, high>={args.pseudo_high}, weight={args.pseudo_weight}, "
            f"real_filter={not args.no_pseudo_real_filter}, real_like={real_like}, synthetic_like={synthetic_like})",
            flush=True,
        )
    y = train.y.astype("float32")
    folds = make_folds(
        y,
        args.folds,
        args.seed,
        strategy=args.fold_strategy,
        angles=train.angles,
        angle_group_decimals=args.angle_group_decimals,
    )
    selected_folds = list(range(args.folds)) if args.fold < 0 else [args.fold]
    if min(selected_folds) < 0 or max(selected_folds) >= args.folds:
        raise ValueError(f"Requested fold outside [0, {args.folds - 1}]: {selected_folds}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.arch}_{args.image_mode}_{args.fold_strategy}_seed{args.seed}_{stamp}"
    oof = np.full(len(y), np.nan, dtype=np.float32)
    test_pred = np.zeros(0 if test is None else len(test.ids), dtype=np.float32)
    histories = []
    fold_metrics = []
    fold_scalers = []

    print(f"device: {device}", flush=True)
    for fold in selected_folds:
        trn_idx, val_idx = folds[fold]
        set_seed(args.seed + fold)
        x_fold_train, x_fold_val, image_scaler = standardize_images(train_channels[trn_idx], train_channels[val_idx])
        aux_fold_train, aux_fold_val, aux_fold_test, aux_scaler = build_aux_features(
            train.images,
            train.angles,
            None if test is None else test.images,
            None if test is None else test.angles,
            trn_idx,
            val_idx,
            args.aux_features,
        )
        fold_scalers.append({"fold": fold, "image_scaler": image_scaler, "aux_scaler": aux_scaler})

        y_fold_train = y[trn_idx]
        fold_weights = np.ones(len(y_fold_train), dtype=np.float32)
        pseudo_count = 0
        if len(pseudo_idx) > 0:
            if test_channels is None or aux_fold_test is None:
                raise RuntimeError("Pseudo-labeling needs test images and auxiliary features.")
            _, x_fold_pseudo, _ = standardize_images(train_channels[trn_idx], test_channels[pseudo_idx])
            x_fold_train = np.concatenate([x_fold_train, x_fold_pseudo], axis=0).astype("float32")
            aux_fold_train = np.concatenate([aux_fold_train, aux_fold_test[pseudo_idx]], axis=0).astype("float32")
            y_fold_train = np.concatenate([y_fold_train, pseudo_y], axis=0).astype("float32")
            fold_weights = np.concatenate(
                [fold_weights, np.full(len(pseudo_y), args.pseudo_weight, dtype=np.float32)],
                axis=0,
            )
            pseudo_count = int(len(pseudo_y))

        train_ds = IcebergDataset(x_fold_train, aux_fold_train, y_fold_train, sample_weights=fold_weights, augment=True)
        val_ds = IcebergDataset(x_fold_val, aux_fold_val, y[val_idx], augment=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)

        model = build_model(
            args.arch,
            in_channels=train_channels.shape[1],
            angle_dim=aux_fold_train.shape[1],
            width=args.width,
            dropout=args.dropout,
            pretrained=not args.no_pretrained,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        best_loss = float("inf")
        best_epoch = -1
        best_path = args.model_dir / f"{prefix}_fold{fold}.pt"
        patience_left = args.patience

        print(
            f"fold {fold}: train={len(trn_idx)} pseudo={pseudo_count} valid={len(val_idx)} "
            f"params={count_parameters(model):,}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_pred = predict(model, val_loader, device, tta=args.val_tta)
            val_loss = binary_log_loss(y[val_idx], val_pred)
            scheduler.step()
            histories.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_log_loss": val_loss,
                    "lr": float(scheduler.get_last_lr()[0]),
                }
            )
            print(f"fold {fold} epoch {epoch:03d}: train={train_loss:.6f} valid={val_loss:.6f}", flush=True)
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_epoch = epoch
                patience_left = args.patience
                torch.save({"model": model.state_dict(), "fold": fold}, best_path)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"fold {fold}: early stopping at epoch {epoch}", flush=True)
                    break

        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        val_pred = predict(model, val_loader, device, tta=args.val_tta)
        oof[val_idx] = val_pred
        fold_metrics.append(
            {"fold": fold, "best_epoch": best_epoch, "best_log_loss": best_loss, "pseudo_count": pseudo_count}
        )

        if test is not None and test_channels is not None and not args.no_test:
            _, x_fold_test, _ = standardize_images(train_channels[trn_idx], test_channels)
            if aux_fold_test is None:
                raise RuntimeError("Test auxiliary features were not built.")
            test_ds = IcebergDataset(x_fold_test, aux_fold_test, y=None, augment=False)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)
            test_pred += predict(model, test_loader, device, tta=args.tta) / len(selected_folds)

    trained_mask = ~np.isnan(oof)
    cv_loss = binary_log_loss(y[trained_mask], oof[trained_mask])
    pd.DataFrame(histories).to_csv(args.report_dir / f"history_{prefix}.csv", index=False)
    pd.DataFrame({"id": train.ids, "is_iceberg": y, "prediction": oof}).to_csv(
        args.output_dir / f"oof_{prefix}.csv", index=False
    )
    if test is not None and not args.no_test:
        submission = pd.DataFrame({"id": test.ids, "is_iceberg": np.clip(test_pred, 1e-5, 1.0 - 1e-5)})
        submission.to_csv(args.output_dir / f"submission_{prefix}.csv", index=False)

    metrics = {
        "prefix": prefix,
        "seed": args.seed,
        "folds": args.folds,
        "trained_folds": list(selected_folds),
        "cv_log_loss_trained_folds": cv_loss,
        "folds_detail": fold_metrics,
        "fold_scalers": fold_scalers,
        "args": vars(args),
    }
    with (args.report_dir / f"metrics_{prefix}.json").open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"CV log_loss on trained folds: {cv_loss:.6f}", flush=True)
    print(f"saved outputs with prefix: {prefix}", flush=True)


if __name__ == "__main__":
    main()
