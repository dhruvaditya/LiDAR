import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Support both direct execution and module import
try:
    from .data.las_dataset import (
        RandomPointBlockDataset,
        SpatiallyRegularDataset,
        compute_class_weights,
        load_points_and_labels,
        normalize_points,
        train_val_split,
    )
    from .models.randlanet import RandLANet
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from randlanet_powerline.data.las_dataset import (
        RandomPointBlockDataset,
        SpatiallyRegularDataset,
        compute_class_weights,
        load_points_and_labels,
        normalize_points,
        train_val_split,
    )
    from randlanet_powerline.models.randlanet import RandLANet


# ---------------------------------------------------------------------------
# Multi-class metric helpers  (replaces the broken binary_metrics pair)
# ---------------------------------------------------------------------------

def confusion_multiclass(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Accumulate a (num_classes, num_classes) confusion matrix on CPU."""
    assert pred.shape == target.shape, f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
    mask     = (target >= 0) & (target < num_classes)
    combined = num_classes * target[mask] + pred[mask]
    cm       = torch.bincount(combined, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)


def multiclass_metrics(cm: torch.Tensor) -> dict:
    """
    Compute per-class and mean IoU / F1 / accuracy from a confusion matrix.
    Returns a dict with keys: iou, mean_f1, accuracy, iou_per_class, f1_per_class.
    """
    cm  = cm.float()
    tp  = cm.diag()
    fn  = cm.sum(dim=1) - tp      # ground-truth positives missed
    fp  = cm.sum(dim=0) - tp      # false alarms

    iou_per_class = tp / (tp + fp + fn).clamp(min=1e-8)
    f1_per_class  = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-8)

    return {
        "iou":           iou_per_class.mean().item(),
        "mean_f1":       f1_per_class.mean().item(),
        "accuracy":      (tp.sum() / cm.sum().clamp(min=1e-8)).item(),
        "iou_per_class": iou_per_class.tolist(),
        "f1_per_class":  f1_per_class.tolist(),
    }


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def validate(model: nn.Module, loader: DataLoader,
             device: torch.device, criterion: nn.Module,
             num_classes: int) -> dict:
    model.eval()
    total_loss = 0.0
    cm_total   = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for xyz, y in loader:
            # Expected shapes: xyz [B, N, 3]  |  y [B, N]
            assert xyz.ndim == 3 and xyz.shape[-1] == 3, \
                f"Expected xyz [B, N, 3], got {xyz.shape}"
            assert y.ndim == 2, \
                f"Expected y [B, N], got {y.shape}"

            xyz = xyz.to(device)
            y   = y.to(device)

            logits = model(xyz)                   # [B, num_classes, N]
            loss   = criterion(logits, y)
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1)    # [B, N]
            cm_total += confusion_multiclass(
                pred.cpu().reshape(-1),
                y.cpu().reshape(-1),
                num_classes,
            )

    metrics         = multiclass_metrics(cm_total)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_dataset_dir(dataset_dir: str) -> str:
    # Use the explicit dataset directory when available.
    if os.path.exists(os.path.join(dataset_dir, "train_randla_net.las")):
        return dataset_dir

    # If the user pointed to the parent `data/` folder, prefer `data/data_split/`.
    candidate = os.path.join(dataset_dir, "data_split")
    if os.path.exists(os.path.join(candidate, "train_randla_net.las")):
        return candidate

    return dataset_dir


def main():
    parser = argparse.ArgumentParser(description="Train RandLA-Net for power-line detection")
    parser.add_argument("--dataset_dir",         type=str,   default="data/data_split")
    parser.add_argument("--epochs",              type=int,   default=40)
    parser.add_argument("--batch_size",          type=int,   default=4)
    parser.add_argument("--num_points",          type=int,   default=4096)
    parser.add_argument("--num_classes",         type=int,   default=3,
                        help="Number of semantic classes – must match your label set")
    parser.add_argument("--train_steps",         type=int,   default=250)
    parser.add_argument("--val_steps",           type=int,   default=50)
    parser.add_argument("--val_ratio",           type=float, default=0.2)
    parser.add_argument("--lr",                  type=float, default=1e-3)
    parser.add_argument("--weight_decay",        type=float, default=1e-4)
    parser.add_argument("--max_points_per_file", type=int,   default=0)
    parser.add_argument("--save_dir",            type=str,   default="checkpoints")
    parser.add_argument("--seed",                type=int,   default=42)
    parser.add_argument("--num_workers",         type=int,   default=0,
                        help="DataLoader workers (0 = single-process, safe default)")
    parser.add_argument("--use_spatial_regular", action="store_true",
                        help="Use spatially regular sampling (original RandLA-Net style)")
    parser.add_argument("--noise_init",          type=float, default=3.5,
                        help="Noise init radius for spatial sampling")
    args = parser.parse_args()
    args.dataset_dir = resolve_dataset_dir(args.dataset_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    max_pts = args.max_points_per_file if args.max_points_per_file > 0 else None

    # -----------------------------------------------------------------------
    # Load & normalise training data
    # -----------------------------------------------------------------------
    print("Loading training data from train_randla_net.las...")
    try:
        x_train, y_train = load_points_and_labels(
            args.dataset_dir,
            file_names=["train_randla_net.las"],
            max_points_per_file=max_pts,
        )
        print(f"  {x_train.shape[0]:,} training points loaded")
    except FileNotFoundError:
        raise FileNotFoundError(f"train_randla_net.las not found in {args.dataset_dir}")

    x_train, center, scale = normalize_points(x_train)

    # -----------------------------------------------------------------------
    # Load (or split) validation data — apply the SAME normalisation
    # -----------------------------------------------------------------------
    val_data_path = os.path.join(args.dataset_dir, "val_randla_net.las")
    if os.path.exists(val_data_path):
        print("Loading validation data from val_randla_net.las...")
        x_val, y_val = load_points_and_labels(
            args.dataset_dir,
            file_names=["val_randla_net.las"],
            max_points_per_file=max_pts,
        )
        x_val = (x_val - center) / max(scale, 1e-8)
        print(f"  {x_val.shape[0]:,} validation points loaded")
    else:
        print("No val_randla_net.las found – using a portion of training data for validation...")
        x_train, y_train, x_val, y_val = train_val_split(
            x_train, y_train,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    # -----------------------------------------------------------------------
    # Sanity-check labels
    # -----------------------------------------------------------------------
    unique_train = np.unique(y_train)
    unique_val   = np.unique(y_val)
    print(f"Label values – train: {unique_train.tolist()}  |  val: {unique_val.tolist()}")
    assert int(unique_train.max()) < args.num_classes, (
        f"Max label {unique_train.max()} >= num_classes {args.num_classes}. "
        "Adjust --num_classes or fix your label encoding."
    )

    # -----------------------------------------------------------------------
    # Datasets & loaders
    # -----------------------------------------------------------------------
    train_ds_kwargs = dict(
        num_points=args.num_points,
        steps_per_epoch=args.train_steps,
        augment=True,
    )
    if args.use_spatial_regular:
        train_ds_kwargs["noise_init"] = args.noise_init
        train_ds = SpatiallyRegularDataset(x_train, y_train, **train_ds_kwargs)
    else:
        train_ds = RandomPointBlockDataset(x_train, y_train, **train_ds_kwargs)

    val_ds = RandomPointBlockDataset(
        x_val, y_val,
        num_points=args.num_points,
        steps_per_epoch=args.val_steps,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # -----------------------------------------------------------------------
    # Model / loss / optimiser / scheduler
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RandLANet(num_classes=args.num_classes).to(device)

    class_weights   = compute_class_weights(y_train, num_classes=args.num_classes)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion       = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou = -1.0

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for xyz, y in train_loader:
            # xyz: [B, N, 3]  |  y: [B, N]
            xyz = xyz.to(device)
            y   = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xyz)           # [B, num_classes, N]
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        train_loss  = running_loss / max(len(train_loader), 1)
        val_metrics = validate(model, val_loader, device, criterion, args.num_classes)

        iou_str = "  ".join(
            f"cls{i}={v:.3f}" for i, v in enumerate(val_metrics["iou_per_class"])
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_mIoU={val_metrics['iou']:.4f} | "
            f"val_mF1={val_metrics['mean_f1']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}\n"
            f"         per-class IoU → {iou_str}"
        )

        ckpt = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "center":               center,       # numpy array – needed at inference
            "scale":                scale,        # scalar    – needed at inference
            "num_classes":          args.num_classes,
            "args":                 vars(args),
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print(f"         → New best mIoU: {best_iou:.4f}  (best.pt saved)")

    print(f"\nTraining complete. Best validation mIoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()