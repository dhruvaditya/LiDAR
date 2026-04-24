#!/usr/bin/env python3
"""
Binary Training Script for Electric Pole & Line Detection with RandLA-Net
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import config and utilities
from config_binary import *
from utils_binary import *

# Import existing modules
try:
    from data.las_dataset import (
        RandomPointBlockDataset,
        SpatiallyRegularDataset,
        load_points_and_labels,
        normalize_points,
    )
    from models.randlanet import RandLANet
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, str(Path(__file__).parent))
    from randlanet_powerline.data.las_dataset import (
        RandomPointBlockDataset,
        SpatiallyRegularDataset,
        load_points_and_labels,
        normalize_points,
    )
    from randlanet_powerline.models.randlanet import RandLANet


# ---------------------------------------------------------------------------
# Loss Functions (adapted for binary)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)  # [B, C, N]

        target_one_hot = F.one_hot(target, num_classes=NUM_CLASSES)  # [B, N, C]
        target_one_hot = target_one_hot.permute(0, 2, 1).float()     # [B, C, N]

        intersection = (pred * target_one_hot).sum(dim=(0, 2))
        pred_sum     = pred.sum(dim=(0, 2))
        target_sum   = target_one_hot.sum(dim=(0, 2))

        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=FOCAL_GAMMA, dice_weight=DICE_WEIGHT):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss  = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice  = self.dice_loss(inputs, targets)
        return (1 - self.dice_weight) * focal + self.dice_weight * dice


# ---------------------------------------------------------------------------
# Validation Function
# ---------------------------------------------------------------------------

def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> Dict[str, float]:
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xyz, y in loader:
            xyz = xyz.to(device)
            y = y.to(device)

            with torch.autocast(device_type=device.type, enabled=USE_AMP):
                logits = model(xyz)
                loss = criterion(logits, y)

            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Compute binary metrics
    metrics = compute_binary_metrics(all_targets, all_preds)
    metrics["loss"] = total_loss / max(len(loader), 1)

    return metrics


# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, scaler: torch.amp.GradScaler,
                gradient_clip: float = 1.0) -> float:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0

    for xyz, y in loader:
        xyz = xyz.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=USE_AMP):
            logits = model(xyz)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train RandLA-Net for binary electric pole/line detection")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint path")
    parser.add_argument("--experiment_name", type=str, default="binary_training",
                        help="Name for this training experiment")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs")
    args = parser.parse_args()

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = setup_device()
    print_config()

    # TensorBoard
    if USE_TENSORBOARD:
        writer = SummaryWriter(TENSORBOARD_DIR / args.experiment_name)
    else:
        writer = None

    # -----------------------------------------------------------------------
    # Load training data
    # -----------------------------------------------------------------------
    print(f"\nLoading training data from: {TRAIN_FILE}")
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}")

    x_train, y_train = load_points_and_labels(
        str(DATA_SPLIT_DIR),
        file_names=[TRAIN_FILE.name]
    )
    print(f"Loaded {x_train.shape[0]:,} training points")

    # Normalize training data
    x_train, center, scale = normalize_points(x_train)
    print(f"Normalization: center={center}, scale={scale:.6f}")

    # -----------------------------------------------------------------------
    # Load validation data
    # -----------------------------------------------------------------------
    print(f"\nLoading validation data from: {VAL_FILE}")
    if not VAL_FILE.exists():
        raise FileNotFoundError(f"Validation file not found: {VAL_FILE}")

    x_val, y_val = load_points_and_labels(
        str(DATA_SPLIT_DIR),
        file_names=[VAL_FILE.name]
    )

    # Apply same normalization as training
    x_val = (x_val - center) / max(scale, 1e-8)
    print(f"Loaded {x_val.shape[0]:,} validation points")

    # -----------------------------------------------------------------------
    # Verify labels
    # -----------------------------------------------------------------------
    unique_train = np.unique(y_train)
    unique_val = np.unique(y_val)
    print(f"Training labels: {sorted(unique_train)}")
    print(f"Validation labels: {sorted(unique_val)}")

    expected_labels = set(range(NUM_CLASSES))
    if not set(unique_train).issubset(expected_labels):
        raise ValueError(f"Training labels {unique_train} not in expected range {expected_labels}")
    if not set(unique_val).issubset(expected_labels):
        raise ValueError(f"Validation labels {unique_val} not in expected range {expected_labels}")

    # -----------------------------------------------------------------------
    # Compute class weights
    # -----------------------------------------------------------------------
    class_weights = compute_binary_class_weights(y_train)
    class_weights_t = torch.tensor(list(class_weights.values()), dtype=torch.float32, device=device)
    print(f"Class weights: {class_weights}")

    # -----------------------------------------------------------------------
    # Create datasets and loaders
    # -----------------------------------------------------------------------
    train_dataset = SpatiallyRegularDataset(
        x_train, y_train,
        num_points=NUM_POINTS,
        steps_per_epoch=STEPS_PER_EPOCH,
        augment=USE_AUGMENTATION
    )

    val_dataset = RandomPointBlockDataset(
        x_val, y_val,
        num_points=NUM_POINTS,
        steps_per_epoch=VAL_STEPS_PER_EPOCH,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS
    )

    # -----------------------------------------------------------------------
    # Model, loss, optimizer, scheduler
    # -----------------------------------------------------------------------
    model = RandLANet(num_classes=NUM_CLASSES).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = CombinedLoss(alpha=class_weights_t, gamma=FOCAL_GAMMA, dice_weight=DICE_WEIGHT)
    print(f"Using Combined Loss (Focal γ={FOCAL_GAMMA} + Dice w={DICE_WEIGHT})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        eps=EPS
    )

    # Learning rate scheduler: warmup + cosine annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - WARMUP_EPOCHS, 1), eta_min=LEARNING_RATE * 1e-2
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS]
    )

    # Mixed precision
    scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP)

    # -----------------------------------------------------------------------
    # Resume training if checkpoint provided
    # -----------------------------------------------------------------------
    start_epoch = 1
    best_val_miou = -1.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_metadata = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint_metadata['epoch'] + 1
        best_val_miou = checkpoint_metadata.get('best_val_miou', -1.0)
        print(f"Resumed from epoch {checkpoint_metadata['epoch']}, best mIoU: {best_val_miou:.4f}")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\nStarting training for {args.epochs} epochs...")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)

        # Validate
        val_metrics = validate(model, val_loader, device, criterion)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        print(f"\nEpoch {epoch:03d}/{EPOCHS} | LR: {current_lr:.2e}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val mIoU: {val_metrics['miou']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Class 0 IoU (Background): {val_metrics['iou_class_0']:.4f}")
        print(f"Class 1 IoU (Electrical): {val_metrics['iou_class_1']:.4f}")

        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Metrics/mIoU', val_metrics['miou'], epoch)
            writer.add_scalar('Metrics/F1', val_metrics['f1'], epoch)
            writer.add_scalar('Metrics/Accuracy', val_metrics['accuracy'], epoch)
            writer.add_scalar('Metrics/IoU_Class_0', val_metrics['iou_class_0'], epoch)
            writer.add_scalar('Metrics/IoU_Class_1', val_metrics['iou_class_1'], epoch)
            writer.add_scalar('LR', current_lr, epoch)

        # Save checkpoints
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            save_checkpoint(
                CHECKPOINTS_DIR / f"binary_epoch_{epoch}.pt",
                model, optimizer, scheduler, epoch, best_val_miou, center, scale
            )

        # Save best model
        if val_metrics['miou'] > best_val_miou:
            best_val_miou = val_metrics['miou']
            patience_counter = 0
            save_checkpoint(
                BEST_CHECKPOINT,
                model, optimizer, scheduler, epoch, best_val_miou, center, scale
            )
            print(f"→ New best mIoU: {best_val_miou:.4f} (saved to {BEST_CHECKPOINT.name})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs ({PATIENCE} without improvement)")
            break

    # Save final model
    save_checkpoint(
        LAST_CHECKPOINT,
        model, optimizer, scheduler, epoch, best_val_miou, center, scale
    )

    print(f"\nTraining complete!")
    print(f"Best validation mIoU: {best_val_miou:.4f}")
    print(f"Checkpoints saved in: {CHECKPOINTS_DIR}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()