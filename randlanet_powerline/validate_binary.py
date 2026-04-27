#!/usr/bin/env python3
"""
Binary Validation Script for Electric Pole & Line Detection

Evaluates a trained model on validation data and prints detailed metrics.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import config and utilities
from config_binary import *
from utils_binary import *

# Import existing modules
try:
    from data.las_dataset import RandomPointBlockDataset, load_points_and_labels, normalize_points
    from models.randlanet import RandLANet
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from randlanet_powerline.data.las_dataset import RandomPointBlockDataset, load_points_and_labels, normalize_points
    from randlanet_powerline.models.randlanet import RandLANet


def main():
    parser = argparse.ArgumentParser(description="Validate binary RandLA-Net model")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_CHECKPOINT),
                        help="Path to model checkpoint")
    parser.add_argument("--val_file", type=str, default=str(VAL_FILE),
                        help="Path to validation LAS file")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for validation")
    parser.add_argument("--num_points", type=int, default=NUM_POINTS,
                        help="Number of points per batch")
    parser.add_argument("--steps", type=int, default=VAL_STEPS_PER_EPOCH,
                        help="Number of validation steps")
    parser.add_argument("--save_confusion_matrix", action="store_true", default=SAVE_CONFUSION_MATRIX,
                        help="Save confusion matrix plot")
    args = parser.parse_args()

    device = setup_device()

    # -----------------------------------------------------------------------
    # Load checkpoint and model
    # -----------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    center = checkpoint['center']
    scale = checkpoint['scale']
    num_classes = checkpoint.get('num_classes', NUM_CLASSES)

    if num_classes != NUM_CLASSES:
        print(f"Warning: Checkpoint has {num_classes} classes, but config has {NUM_CLASSES}")

    model = RandLANet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Model loaded successfully")

    # -----------------------------------------------------------------------
    # Load validation data
    # -----------------------------------------------------------------------
    print(f"Loading validation data: {args.val_file}")
    if not Path(args.val_file).exists():
        raise FileNotFoundError(f"Validation file not found: {args.val_file}")

    x_val, y_val = load_points_and_labels(
        str(Path(args.val_file).parent),
        file_names=[Path(args.val_file).name]
    )

    # Apply same normalization as training
    x_val = (x_val - center) / max(scale, 1e-8)

    print(f"Loaded {x_val.shape[0]:,} validation points")
    print(f"Normalization applied: center={center}, scale={scale:.6f}")

    # -----------------------------------------------------------------------
    # Create validation dataset and loader
    # -----------------------------------------------------------------------
    val_dataset = RandomPointBlockDataset(
        x_val, y_val,
        num_points=args.num_points,
        steps_per_epoch=args.steps,
        augment=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    print(f"\nRunning validation ({args.steps} steps)...")

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (xyz, y) in enumerate(val_loader):
            if i >= args.steps:
                break

            xyz = xyz.to(device)
            batch_targets = y.numpy()

            with torch.autocast(device_type=device.type, enabled=USE_AMP):
                logits = model(xyz)

            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.append(batch_preds)
            all_targets.append(batch_targets)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{args.steps} batches")

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    print(f"\nValidation complete!")
    print(f"Total predictions: {len(all_preds):,}")

    # -----------------------------------------------------------------------
    # Compute and display metrics
    # -----------------------------------------------------------------------
    metrics = compute_binary_metrics(all_targets, all_preds)
    print_binary_metrics(metrics)

    # -----------------------------------------------------------------------
    # Save confusion matrix
    # -----------------------------------------------------------------------
    if args.save_confusion_matrix:
        confusion_path = RESULTS_DIR / "validation_confusion_matrix.png"
        plot_confusion_matrix_binary(all_targets, all_preds, save_path=str(confusion_path))

    # -----------------------------------------------------------------------
    # Class distribution
    # -----------------------------------------------------------------------
    print("\nPrediction Distribution:")
    unique_pred, counts_pred = np.unique(all_preds, return_counts=True)
    for cls, count in zip(unique_pred, counts_pred):
        percentage = (count / len(all_preds)) * 100
        class_name = CLASS_NAMES[int(cls)]
        print(f"  Predicted {cls} ({class_name}): {count:,} ({percentage:.1f}%)")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "validation_file": args.val_file,
        "checkpoint": args.checkpoint,
        "total_samples": len(all_preds),
        "metrics": metrics,
        "class_distribution": {
            "predictions": {int(k): int(v) for k, v in zip(unique_pred, counts_pred)},
            "targets": {int(k): int(v) for k, v in np.unique(all_targets, return_counts=True)}
        }
    }

    results_file = RESULTS_DIR / "validation_results.json"
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()