#!/usr/bin/env python3
"""
Utility functions for Binary Electric Pole & Line Detection
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import laspy
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Import config
from config_binary import *


def load_binary_las(file_path: str, max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load binary LAS file and return normalized points and labels.

    Args:
        file_path: Path to LAS file
        max_points: Maximum points to load (random sampling)

    Returns:
        points: Normalized points (N, 3)
        labels: Binary labels (N,)
        center: Center for denormalization
        scale: Scale for denormalization
    """
    las = laspy.read(file_path)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    labels = np.array(las.classification).astype(np.uint8)

    if max_points is not None and xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], max_points, replace=False)
        xyz = xyz[idx]
        labels = labels[idx]

    # Normalize
    from data.las_dataset import normalize_points
    normalized_points, center, scale = normalize_points(xyz)

    return normalized_points, labels, center, scale


def compute_binary_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for binary classification.

    Args:
        labels: Binary labels array

    Returns:
        Dictionary mapping class to weight
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()

    # Inverse frequency weighting
    weights = total / (len(unique) * counts)
    weights = weights / weights.sum() * len(unique)  # Normalize

    return {int(cls): float(w) for cls, w in zip(unique, weights)}


def plot_confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray,
                               save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix for binary classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Labels
    classes = CLASS_NAMES
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")

    ax.set_title('Binary Classification Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()


def plot_class_distribution(train_labels: np.ndarray, val_labels: np.ndarray,
                          test_labels: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Plot class distribution across splits.

    Args:
        train_labels, val_labels, test_labels: Label arrays for each split
        save_path: Path to save plot (optional)
    """
    splits = ['Train', 'Val', 'Test']
    labels_data = [train_labels, val_labels, test_labels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (split, labels) in enumerate(zip(splits, labels_data)):
        unique, counts = np.unique(labels, return_counts=True)
        percentages = (counts / counts.sum()) * 100

        bars = axes[i].bar(range(len(unique)), percentages)
        axes[i].set_title(f'{split} Set')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel('Percentage (%)')
        axes[i].set_xticks(range(len(unique)))
        axes[i].set_xticklabels([CLASS_NAMES[int(cls)] for cls in unique])

        # Add value labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{pct:.1f}%', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to: {save_path}")
    else:
        plt.show()


def save_predictions_to_las(points: np.ndarray, predictions: np.ndarray,
                          center: np.ndarray, scale: float,
                          output_path: str, confidence: Optional[np.ndarray] = None) -> None:
    """
    Save predictions to LAS file.

    Args:
        points: Normalized points (N, 3)
        predictions: Predicted classes (N,)
        center: Center for denormalization
        scale: Scale for denormalization
        output_path: Output LAS file path
        confidence: Prediction confidence scores (optional)
    """
    # Denormalize points
    denormalized = points * scale + center

    # Create LAS file
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="predicted_class", type=np.uint8))

    if confidence is not None:
        header.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.float32))

    las = laspy.LasData(header)

    # Set coordinates
    las.x = denormalized[:, 0]
    las.y = denormalized[:, 1]
    las.z = denormalized[:, 2]

    # Set predictions
    las.classification = predictions.astype(np.uint8)

    if confidence is not None:
        las.confidence = confidence.astype(np.float32)

    # Save
    las.write(output_path)
    print(f"Predictions saved to: {output_path}")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[object] = None) -> Dict:
    """
    Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)

    Returns:
        Checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'best_val_miou': checkpoint.get('best_val_miou', 0.0),
        'center': checkpoint.get('center'),
        'scale': checkpoint.get('scale'),
        'num_classes': checkpoint.get('num_classes'),
        'class_names': checkpoint.get('class_names')
    }

    return metadata


def save_checkpoint(checkpoint_path: str, model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer, scheduler: object,
                   epoch: int, best_val_miou: float,
                   center: np.ndarray, scale: float) -> None:
    """
    Save model checkpoint.

    Args:
        checkpoint_path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Scheduler to save
        epoch: Current epoch
        best_val_miou: Best validation mIoU
        center: Normalization center
        scale: Normalization scale
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_miou': best_val_miou,
        'center': center,
        'scale': scale,
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Basic metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # IoU (Jaccard index)
    iou_class_0 = tn / (tn + fn + fp) if (tn + fn + fp) > 0 else 0.0
    iou_class_1 = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0
    miou = (iou_class_0 + iou_class_1) / 2

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou_class_0': iou_class_0,
        'iou_class_1': iou_class_1,
        'miou': miou,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def print_binary_metrics(metrics: Dict[str, float]) -> None:
    """Print binary classification metrics in a nice format."""
    print("\nBinary Classification Metrics:")
    print("-" * 40)
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print("-" * 40)
    print(f"Confusion Matrix: TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}")


def setup_device() -> torch.device:
    """Setup compute device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return {'allocated_gb': allocated, 'reserved_gb': reserved}
    else:
        return {'allocated_gb': 0.0, 'reserved_gb': 0.0}