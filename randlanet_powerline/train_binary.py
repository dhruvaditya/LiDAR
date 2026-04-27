#!/usr/bin/env python3
"""
Binary Training Script for Electric Pole & Line Detection with RandLA-Net
=========================================================================
Key improvements over original:
  1. Focal + Dice combined loss  (handles 47:1 class imbalance)
  2. Guaranteed minority oversampling in every batch
  3. Aggressive class weight boosting for wire class
  4. Threshold-tuned inference  (not just argmax)
  5. Per-epoch class-1 recall printed so you can see collapse early
  6. Full metric suite: IoU per class, F1, Precision, Recall, Accuracy
  7. Warmup + cosine LR schedule
  8. Gradient clipping
  9. Mixed-precision (AMP) training
  10. Early stopping with patience
  11. TensorBoard logging
  12. Clean checkpoint save / resume
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

# ---------------------------------------------------------------------------
# Path resolution — works both as standalone and inside a package
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE, _HERE / "randlanet_powerline"]:
    if (_candidate / "data").exists():
        sys.path.insert(0, str(_candidate))
        break

from data.las_dataset import (
    RandomPointBlockDataset,
    SpatiallyRegularDataset,
    load_points_and_labels,
    normalize_points,
)
from models.randlanet import RandLANet


# ===========================================================================
# ▌ CONFIG  (edit these — replaces config_binary.py)
# ===========================================================================

# --- Paths -----------------------------------------------------------------
DATA_SPLIT_DIR  = _HERE / "data" / "data_split"
TRAIN_FILE_NAME = "train_binary.las"
VAL_FILE_NAME   = "val_binary.las"
CHECKPOINTS_DIR = _HERE / "checkpoints"
TENSORBOARD_DIR = _HERE / "runs"
RESULTS_DIR     = _HERE / "results"

# --- Model -----------------------------------------------------------------
NUM_CLASSES = 2      # 0 = background, 1 = electrical poles/lines

# --- Training --------------------------------------------------------------
EPOCHS              = 100
BATCH_SIZE          = 4
NUM_POINTS          = 4096
STEPS_PER_EPOCH     = 300
VAL_STEPS_PER_EPOCH = 60
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-4
BETAS               = (0.9, 0.999)
EPS                 = 1e-8
WARMUP_EPOCHS       = 8
SEED                = 42
NUM_WORKERS         = 0        # 0 = safe on Windows; increase on Linux
PIN_MEMORY          = True
PERSISTENT_WORKERS  = False    # requires NUM_WORKERS > 0

# --- Loss ------------------------------------------------------------------
FOCAL_GAMMA   = 2.5   # higher γ = more focus on hard/rare examples
DICE_WEIGHT   = 0.4   # combined = (1-DICE_WEIGHT)*focal + DICE_WEIGHT*dice

# --- Class weight amplifier ------------------------------------------------
# Inverse-frequency gives ~47× for wire class on typical data.
# This multiplier pushes it further if recall stays near zero.
WIRE_CLASS_WEIGHT_BOOST = 3.0   # set to 1.0 to disable; try 3–10

# --- Oversampling ----------------------------------------------------------
# Fraction of each batch guaranteed to be wire (class-1) points
MIN_WIRE_RATIO = 0.15            # 0.0 = disable balanced sampling

# --- Inference threshold ---------------------------------------------------
# Use lower threshold than 0.5 to catch rare wire points
WIRE_DECISION_THRESHOLD = 0.30  # reduce to 0.20 if recall is still ~0

# --- Augmentation ----------------------------------------------------------
USE_AUGMENTATION = True

# --- Regularisation --------------------------------------------------------
GRADIENT_CLIP = 1.0
USE_AMP       = torch.cuda.is_available()   # mixed precision (GPU only)

# --- Checkpointing ---------------------------------------------------------
SAVE_INTERVAL    = 10    # save every N epochs
PATIENCE         = 20    # early-stopping patience (epochs without IoU gain)
USE_TENSORBOARD  = HAS_TENSORBOARD


# ===========================================================================
# ▌ BALANCED DATASET  (guarantees wire points in every batch)
# ===========================================================================

class BalancedBlockDataset(Dataset):
    """
    Wraps the raw point arrays and ensures each sampled block contains at
    least `min_wire_ratio` fraction of class-1 (wire) points.

    If min_wire_ratio == 0.0, falls back to purely random sampling
    (same behaviour as RandomPointBlockDataset).
    """

    def __init__(
        self,
        xyz:             np.ndarray,
        labels:          np.ndarray,
        num_points:      int   = 4096,
        steps_per_epoch: int   = 300,
        min_wire_ratio:  float = 0.15,
        augment:         bool  = True,
    ) -> None:
        self.xyz             = xyz.astype(np.float32)
        self.labels          = labels.astype(np.int64)
        self.num_points      = num_points
        self.steps_per_epoch = steps_per_epoch
        self.min_wire_ratio  = min_wire_ratio
        self.augment         = augment

        self.wire_idx = np.where(labels == 1)[0]
        self.bg_idx   = np.where(labels == 0)[0]

        if len(self.wire_idx) == 0:
            raise RuntimeError(
                "No class-1 (wire) points found in this split!\n"
                "Check that your LAS classification field contains label=1."
            )

        print(f"  BalancedDataset | wire={len(self.wire_idx):,}  "
              f"bg={len(self.bg_idx):,}  "
              f"ratio={len(self.wire_idx)/max(len(self.bg_idx),1):.4f}")

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, _idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.min_wire_ratio > 0.0 and len(self.wire_idx) > 0:
            n_wire = max(1, int(self.num_points * self.min_wire_ratio))
            n_bg   = self.num_points - n_wire
            wire_chosen = np.random.choice(
                self.wire_idx, n_wire, replace=(len(self.wire_idx) < n_wire)
            )
            bg_chosen = np.random.choice(
                self.bg_idx, n_bg, replace=(len(self.bg_idx) < n_bg)
            )
            chosen = np.concatenate([wire_chosen, bg_chosen])
            np.random.shuffle(chosen)
        else:
            chosen = np.random.choice(len(self.xyz), self.num_points, replace=False)

        xyz    = self.xyz[chosen].copy()
        labels = self.labels[chosen].copy()

        if self.augment:
            # Random Z-axis rotation
            angle = np.random.uniform(0, 2 * np.pi)
            c, s  = np.cos(angle), np.sin(angle)
            R     = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            xyz   = xyz @ R.T
            # Gaussian jitter
            xyz  += np.random.normal(0, 0.005, xyz.shape).astype(np.float32)
            # Random scale
            xyz  *= np.random.uniform(0.95, 1.05)
            # Random flip X
            if np.random.rand() < 0.5:
                xyz[:, 0] *= -1

        return (
            torch.from_numpy(xyz),     # [N, 3]
            torch.from_numpy(labels),  # [N]
        )


# ===========================================================================
# ▌ LOSS FUNCTIONS
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Focal loss — down-weights the easy majority-class examples so the
    gradient is dominated by the hard, rare wire points.

    gamma=0 → standard cross-entropy
    gamma=2 → standard focal (paper default)
    gamma=2.5 → more aggressive; useful at 47:1 imbalance
    """

    def __init__(self, alpha=None, gamma: float = 2.5, reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha      # per-class weight tensor [C]
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [B, C, N]   targets: [B, N]
        ce_loss    = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt         = torch.exp(-ce_loss)                        # P(correct class)
        focal_loss = (1 - pt) ** self.gamma * ce_loss           # reweight

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class SoftDiceLoss(nn.Module):
    """
    Soft Dice loss — directly optimises the F1/Dice overlap coefficient.
    Complements focal loss: focal drives recall, dice prevents precision collapse.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [B, C, N]   targets: [B, N]
        probs = F.softmax(inputs, dim=1)                         # [B, C, N]

        # One-hot encode targets
        B, C, N = probs.shape
        target_oh = F.one_hot(targets, num_classes=C)            # [B, N, C]
        target_oh = target_oh.permute(0, 2, 1).float()           # [B, C, N]

        # Per-class Dice
        intersection = (probs * target_oh).sum(dim=(0, 2))       # [C]
        denom        = probs.sum(dim=(0, 2)) + target_oh.sum(dim=(0, 2))  # [C]
        dice_per_cls = (2.0 * intersection + self.smooth) / (denom + self.smooth)

        return 1.0 - dice_per_cls.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss = (1 - dice_weight) * FocalLoss + dice_weight * DiceLoss

    This is the most effective single change for severe class imbalance.
    Focal handles point-level recall; Dice handles region-level overlap.
    """

    def __init__(
        self,
        alpha=None,
        gamma: float = 2.5,
        dice_weight: float = 0.4,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice  = SoftDiceLoss()
        self.dw    = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.dw) * self.focal(inputs, targets) + \
               self.dw         * self.dice(inputs, targets)


# ===========================================================================
# ▌ METRICS
# ===========================================================================

def compute_metrics(
    targets: np.ndarray,
    preds:   np.ndarray,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Full metric suite from flat target/pred arrays.

    Returns dict with keys:
        miou, f1, accuracy, precision, recall,
        iou_class_0, iou_class_1,
        f1_class_0,  f1_class_1,
        precision_class_1, recall_class_1
    """
    metrics: Dict[str, float] = {}

    iou_list, f1_list = [], []
    for c in range(num_classes):
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        tn = int(((preds != c) & (targets != c)).sum())

        iou = tp / max(tp + fp + fn, 1)
        f1  = 2 * tp / max(2 * tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)

        iou_list.append(iou)
        f1_list.append(f1)
        metrics[f"iou_class_{c}"]       = iou
        metrics[f"f1_class_{c}"]        = f1
        metrics[f"precision_class_{c}"] = prec
        metrics[f"recall_class_{c}"]    = rec

    metrics["miou"]      = float(np.mean(iou_list))
    metrics["f1"]        = float(np.mean(f1_list))
    metrics["accuracy"]  = float((preds == targets).sum() / max(len(targets), 1))
    metrics["precision"] = metrics["precision_class_1"]
    metrics["recall"]    = metrics["recall_class_1"]

    return metrics


def compute_class_weights(labels: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Inverse-frequency class weights, then boost the wire class further by
    WIRE_CLASS_WEIGHT_BOOST to push the model toward higher recall.
    """
    counts  = np.bincount(labels.ravel(), minlength=num_classes).astype(np.float64)
    counts  = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights /= weights.sum()
    weights *= num_classes

    # Amplify the minority (wire) class beyond inverse-frequency
    weights[1] *= WIRE_CLASS_WEIGHT_BOOST
    weights /= weights.mean()          # keep overall scale stable

    return weights.astype(np.float32)


# ===========================================================================
# ▌ CHECKPOINT HELPERS
# ===========================================================================

def save_checkpoint(
    path:        Path,
    model:       nn.Module,
    optimizer,
    scheduler,
    epoch:       int,
    best_miou:   float,
    center:      np.ndarray,
    scale:       float,
    threshold:   float,
) -> None:
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_miou":        best_miou,
        "center":               center,
        "scale":                scale,
        "wire_threshold":       threshold,
        "num_classes":          NUM_CLASSES,
    }, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# ===========================================================================
# ▌ VALIDATION LOOP
# ===========================================================================

def validate(
    model:     nn.Module,
    loader:    DataLoader,
    device:    torch.device,
    criterion: nn.Module,
    threshold: float = 0.30,
) -> Dict[str, float]:
    """
    Validate model. Uses `threshold` instead of argmax so rare wire points
    are not systematically suppressed by a 0.5 decision boundary.
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xyz, y in loader:
            xyz = xyz.to(device)
            y   = y.to(device)

            with torch.autocast(device_type=device.type, enabled=USE_AMP):
                logits = model(xyz)                       # [B, C, N]
                loss   = criterion(logits, y)

            total_loss += loss.item()

            # Threshold-based prediction (not argmax)
            probs      = torch.softmax(logits, dim=1)    # [B, C, N]
            wire_prob  = probs[:, 1, :]                  # [B, N]
            pred       = (wire_prob > threshold).long()  # [B, N]

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_preds   = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel()

    metrics         = compute_metrics(all_targets, all_preds, NUM_CLASSES)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


# ===========================================================================
# ▌ TRAINING LOOP
# ===========================================================================

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler:    torch.amp.GradScaler,
) -> float:
    model.train()
    total_loss = 0.0

    for xyz, y in loader:
        xyz = xyz.to(device)
        y   = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=USE_AMP):
            logits = model(xyz)                  # [B, C, N]
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()

        if GRADIENT_CLIP > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ===========================================================================
# ▌ MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RandLA-Net (binary) for electric pole/line detection"
    )
    parser.add_argument("--resume",          type=str,   default=None)
    parser.add_argument("--experiment_name", type=str,   default="binary_training")
    parser.add_argument("--epochs",          type=int,   default=EPOCHS)
    parser.add_argument("--lr",              type=float, default=LEARNING_RATE)
    parser.add_argument("--threshold",       type=float, default=WIRE_DECISION_THRESHOLD,
                        help="Wire class probability threshold at inference (default 0.30)")
    parser.add_argument("--wire_boost",      type=float, default=WIRE_CLASS_WEIGHT_BOOST,
                        help="Multiply wire class weight by this factor beyond inverse-freq")
    parser.add_argument("--min_wire_ratio",  type=float, default=MIN_WIRE_RATIO,
                        help="Minimum fraction of wire points per batch (0=disabled)")
    parser.add_argument("--gamma",           type=float, default=FOCAL_GAMMA,
                        help="Focal loss gamma (higher = more focus on hard examples)")
    parser.add_argument("--use_tensorboard", action="store_true", default=USE_TENSORBOARD)
    args = parser.parse_args()

    # Override globals from CLI
    wire_boost     = args.wire_boost
    threshold      = args.threshold
    min_wire_ratio = args.min_wire_ratio

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"AMP    : {USE_AMP}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.use_tensorboard and HAS_TENSORBOARD:
        writer = SummaryWriter(TENSORBOARD_DIR / args.experiment_name)
        print(f"TensorBoard: {TENSORBOARD_DIR / args.experiment_name}")

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    train_path = DATA_SPLIT_DIR / TRAIN_FILE_NAME
    val_path   = DATA_SPLIT_DIR / VAL_FILE_NAME

    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    print(f"\nLoading training data: {train_path}")
    x_train, y_train = load_points_and_labels(
        str(DATA_SPLIT_DIR), file_names=[TRAIN_FILE_NAME]
    )
    x_train, center, scale = normalize_points(x_train)
    print(f"  {len(x_train):,} points | center={center} | scale={scale:.6f}")

    print(f"Loading validation data: {val_path}")
    x_val, y_val = load_points_and_labels(
        str(DATA_SPLIT_DIR), file_names=[VAL_FILE_NAME]
    )
    x_val = (x_val - center) / max(float(scale), 1e-8)
    print(f"  {len(x_val):,} points")

    # -----------------------------------------------------------------------
    # Label verification & distribution
    # -----------------------------------------------------------------------
    for split_name, labels in [("train", y_train), ("val", y_val)]:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n{split_name} label distribution:")
        for cls, cnt in zip(unique, counts):
            tag = "wire" if cls == 1 else "background"
            print(f"  class {cls} ({tag}): {cnt:,}  ({100*cnt/len(labels):.2f}%)")
        if not set(unique).issubset(set(range(NUM_CLASSES))):
            raise ValueError(
                f"{split_name} contains unexpected labels {unique}. "
                f"Expected subset of {list(range(NUM_CLASSES))}."
            )

    # -----------------------------------------------------------------------
    # Class weights (inverse-frequency + manual wire boost)
    # -----------------------------------------------------------------------
    class_weights           = compute_class_weights(y_train, NUM_CLASSES)
    class_weights[1]       *= wire_boost
    class_weights           = class_weights / class_weights.mean()
    class_weights_t         = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"\nClass weights (after ×{wire_boost:.1f} wire boost):")
    print(f"  background={class_weights[0]:.3f}  wire={class_weights[1]:.3f}")

    # -----------------------------------------------------------------------
    # Datasets and loaders
    # -----------------------------------------------------------------------
    train_dataset = BalancedBlockDataset(
        x_train, y_train,
        num_points=NUM_POINTS,
        steps_per_epoch=STEPS_PER_EPOCH,
        min_wire_ratio=min_wire_ratio,
        augment=USE_AUGMENTATION,
    )
    val_dataset = BalancedBlockDataset(
        x_val, y_val,
        num_points=NUM_POINTS,
        steps_per_epoch=VAL_STEPS_PER_EPOCH,
        min_wire_ratio=0.0,      # validation: natural distribution
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = RandLANet(num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: RandLANet  |  trainable params: {n_params:,}")

    # -----------------------------------------------------------------------
    # Loss, optimiser, scheduler
    # -----------------------------------------------------------------------
    criterion = CombinedLoss(
        alpha=class_weights_t,
        gamma=args.gamma,
        dice_weight=DICE_WEIGHT,
    )
    print(f"Loss: Focal(γ={args.gamma}) × {1-DICE_WEIGHT:.1f}  +  Dice × {DICE_WEIGHT:.1f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        eps=EPS,
    )

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - WARMUP_EPOCHS, 1),
        eta_min=args.lr * 1e-2,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS]
    )

    scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP)

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    start_epoch   = 1
    best_val_miou = -1.0

    if args.resume and Path(args.resume).exists():
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_miou = ckpt.get("best_val_miou", -1.0)
        threshold     = ckpt.get("wire_threshold", threshold)
        print(f"Resumed from epoch {start_epoch - 1}  |  best_mIoU={best_val_miou:.4f}")

    # -----------------------------------------------------------------------
    # Paths for best / last checkpoints
    # -----------------------------------------------------------------------
    best_ckpt = CHECKPOINTS_DIR / "best.pt"
    last_ckpt = CHECKPOINTS_DIR / "last.pt"

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs  |  threshold={threshold}")
    print(f"{'='*60}\n")

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = validate(model, val_loader, device, criterion, threshold=threshold)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Console log (no broken f-strings) ----------------------------
        sep = "-" * 56
        print(sep)
        print(f"Epoch {epoch:03d}/{args.epochs}  |  LR: {current_lr:.3e}")
        print(sep)
        print(f"  train_loss           : {train_loss:.4f}")
        print(f"  val_loss             : {val_metrics['loss']:.4f}")
        print(sep)
        print(f"  val_mIoU             : {val_metrics['miou']:.4f}")
        print(f"  val_F1  (mean)       : {val_metrics['f1']:.4f}")
        print(f"  val_Accuracy         : {val_metrics['accuracy']:.4f}")
        print(sep)
        print(f"  IoU  background (0)  : {val_metrics['iou_class_0']:.4f}")
        print(f"  IoU  wire       (1)  : {val_metrics['iou_class_1']:.4f}  ← watch this")
        print(f"  F1   wire       (1)  : {val_metrics['f1_class_1']:.4f}")
        print(f"  Prec wire       (1)  : {val_metrics['precision_class_1']:.4f}")
        print(f"  Rec  wire       (1)  : {val_metrics['recall_class_1']:.4f}  ← near 0 = collapse")

        # Collapse early warning
        if val_metrics["recall_class_1"] < 0.05 and epoch >= WARMUP_EPOCHS:
            print("  [!] Wire recall near zero — model may be collapsing.")
            print("      Try: --wire_boost 10  --min_wire_ratio 0.25  --gamma 3.0")

        # --- TensorBoard --------------------------------------------------
        if writer:
            writer.add_scalar("Loss/train",           train_loss,                       epoch)
            writer.add_scalar("Loss/val",             val_metrics["loss"],              epoch)
            writer.add_scalar("Metrics/mIoU",         val_metrics["miou"],              epoch)
            writer.add_scalar("Metrics/F1",           val_metrics["f1"],                epoch)
            writer.add_scalar("Metrics/Accuracy",     val_metrics["accuracy"],          epoch)
            writer.add_scalar("Metrics/IoU_bg",       val_metrics["iou_class_0"],       epoch)
            writer.add_scalar("Metrics/IoU_wire",     val_metrics["iou_class_1"],       epoch)
            writer.add_scalar("Metrics/Recall_wire",  val_metrics["recall_class_1"],    epoch)
            writer.add_scalar("Metrics/Prec_wire",    val_metrics["precision_class_1"], epoch)
            writer.add_scalar("LR",                   current_lr,                       epoch)

        # --- Periodic checkpoint ------------------------------------------
        if epoch % SAVE_INTERVAL == 0 or epoch == args.epochs:
            save_checkpoint(
                CHECKPOINTS_DIR / f"epoch_{epoch:03d}.pt",
                model, optimizer, scheduler, epoch, best_val_miou, center, scale, threshold
            )

        # --- Best checkpoint ----------------------------------------------
        if val_metrics["miou"] > best_val_miou:
            best_val_miou    = val_metrics["miou"]
            patience_counter = 0
            save_checkpoint(
                best_ckpt, model, optimizer, scheduler,
                epoch, best_val_miou, center, scale, threshold
            )
            print(f"\n  → New best mIoU: {best_val_miou:.4f}  (best.pt saved)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping: no improvement for {PATIENCE} epochs.")
                break

    # --- Final checkpoint -------------------------------------------------
    save_checkpoint(
        last_ckpt, model, optimizer, scheduler,
        epoch, best_val_miou, center, scale, threshold
    )

    print(f"\n{'='*60}")
    print(f"Training complete  |  Best validation mIoU: {best_val_miou:.4f}")
    print(f"Best checkpoint  : {best_ckpt}")
    print(f"Last checkpoint  : {last_ckpt}")
    print(f"{'='*60}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()