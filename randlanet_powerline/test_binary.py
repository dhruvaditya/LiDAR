#!/usr/bin/env python3
"""
Binary Testing Script for Electric Pole & Line Detection

Evaluates a trained model on test data, saves predictions, and computes final metrics.
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
    parser = argparse.ArgumentParser(description="Test binary RandLA-Net model on test data")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_CHECKPOINT),
                        help="Path to model checkpoint")
    parser.add_argument("--test_file", type=str, default=str(TEST_FILE),
                        help="Path to test LAS file")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for testing")
    parser.add_argument("--num_points", type=int, default=NUM_POINTS,
                        help="Number of points per batch")
    parser.add_argument("--save_predictions", action="store_true", default=True,
                        help="Save predictions to LAS file")
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
    # Load test data
    # -----------------------------------------------------------------------
    print(f"Loading test data: {args.test_file}")
    if not Path(args.test_file).exists():
        raise FileNotFoundError(f"Test file not found: {args.test_file}")

    # Load original test data (before normalization)
    x_test_orig, y_test_orig = load_points_and_labels(
        str(Path(args.test_file).parent),
        file_names=[Path(args.test_file).name]
    )

    # Apply same normalization as training
    x_test = (x_test_orig - center) / max(scale, 1e-8)

    print(f"Loaded {x_test.shape[0]:,} test points")
    print(f"Normalization applied: center={center}, scale={scale:.6f}")

    # -----------------------------------------------------------------------
    # Create test dataset and loader
    # -----------------------------------------------------------------------
    # For testing, we want to evaluate on the entire dataset
    # Create a dataset that covers all points
    test_dataset = RandomPointBlockDataset(
        x_test, y_test_orig,  # Use original labels for evaluation
        num_points=args.num_points,
        steps_per_epoch=int(np.ceil(x_test.shape[0] / args.num_points)),  # Cover all points
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # -----------------------------------------------------------------------
    # Testing
    # -----------------------------------------------------------------------
    print(f"\nRunning testing on {len(test_dataset)} batches...")

    all_preds = []
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for i, (xyz, y) in enumerate(test_loader):
            xyz = xyz.to(device)

            with torch.autocast(device_type=device.type, enabled=USE_AMP):
                logits = model(xyz)

            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            batch_logits = logits.cpu().numpy()

            all_preds.append(batch_preds)
            all_targets.append(y.numpy())
            all_logits.append(batch_logits)

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_loader)} batches")

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_logits = np.concatenate(all_logits, axis=0)  # Shape: (total_batches * batch_size, num_classes, num_points)

    print(f"\nTesting complete!")
    print(f"Total predictions: {len(all_preds):,}")

    # -----------------------------------------------------------------------
    # Compute and display metrics
    # -----------------------------------------------------------------------
    metrics = compute_binary_metrics(all_targets, all_preds)
    print_binary_metrics(metrics)

    # -----------------------------------------------------------------------
    # Save predictions to LAS file
    # -----------------------------------------------------------------------
    if args.save_predictions:
        # Get confidence scores (probability of class 1)
        confidence_scores = None
        if all_logits.shape[1] == NUM_CLASSES:
            # Apply softmax to get probabilities
            softmax_logits = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
            confidence_scores = softmax_logits[:, 1, :]  # Probability of class 1
            confidence_scores = confidence_scores.flatten()[:len(x_test_orig)]  # Trim to match data size

        # Trim predictions to match original data size
        final_preds = all_preds[:x_test_orig.shape[0]]
        final_confidence = confidence_scores[:x_test_orig.shape[0]] if confidence_scores is not None else None

        predictions_file = TEST_PREDICTIONS_FILE
        save_predictions_to_las(
            x_test_orig[:len(final_preds)],  # Use original (unnormalized) coordinates
            final_preds,
            center, scale,
            str(predictions_file),
            final_confidence
        )

    # -----------------------------------------------------------------------
    # Save confusion matrix
    # -----------------------------------------------------------------------
    if args.save_confusion_matrix:
        confusion_path = RESULTS_DIR / "test_confusion_matrix.png"
        plot_confusion_matrix_binary(all_targets, all_preds, save_path=str(confusion_path))

    # -----------------------------------------------------------------------
    # Class distribution analysis
    # -----------------------------------------------------------------------
    print("\nTest Set Distribution:")
    unique_target, counts_target = np.unique(all_targets, return_counts=True)
    unique_pred, counts_pred = np.unique(all_preds, return_counts=True)

    print("Targets:")
    for cls, count in zip(unique_target, counts_target):
        percentage = (count / len(all_targets)) * 100
        class_name = CLASS_NAMES[int(cls)]
        print(f"  {cls} ({class_name}): {count:,} ({percentage:.1f}%)")

    print("Predictions:")
    for cls, count in zip(unique_pred, counts_pred):
        percentage = (count / len(all_preds)) * 100
        class_name = CLASS_NAMES[int(cls)]
        print(f"  {cls} ({class_name}): {count:,} ({percentage:.1f}%)")

    # -----------------------------------------------------------------------
    # Save detailed results
    # -----------------------------------------------------------------------
    results = {
        "test_file": args.test_file,
        "checkpoint": args.checkpoint,
        "total_samples": len(all_preds),
        "metrics": metrics,
        "class_distribution": {
            "targets": {int(k): int(v) for k, v in zip(unique_target, counts_target)},
            "predictions": {int(k): int(v) for k, v in zip(unique_pred, counts_pred)}
        },
        "predictions_saved": args.save_predictions,
        "predictions_file": str(TEST_PREDICTIONS_FILE) if args.save_predictions else None
    }

    import json
    with open(TEST_RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {TEST_RESULTS_FILE}")

    # Summary
    print("" + "="*60)
    print("TESTING SUMMARY")
    print("="*60)
    print(f"Model: {Path(args.checkpoint).name}")
    print(f"Test Data: {Path(args.test_file).name}")
    print(f"Samples: {len(all_preds):,}")
    print(f"mIoU: {metrics['miou']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Electrical IoU: {metrics['iou_class_1']:.4f}")
    if args.save_predictions:
        print(f"Predictions: {TEST_PREDICTIONS_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()