#!/usr/bin/env python3
"""
Binary Dataset Split for Electric Pole & Line Detection

Creates train/val/test splits from the merged binary LAS dataset.
Uses spatial splitting (by X coordinate) to maintain spatial coherence.

Usage:
    python prepare_binary_split.py
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import laspy
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.las_dataset import normalize_points


def load_binary_las(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load binary LAS file and return normalized points and labels.

    Returns:
        points: Normalized points (N, 3)
        labels: Binary labels (N,)
        center: Center for denormalization
        scale: Scale for denormalization
    """
    las = laspy.read(file_path)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    labels = np.array(las.classification).astype(np.uint8)

    # Normalize
    normalized_points, center, scale = normalize_points(xyz)

    return normalized_points, labels, center, scale


def spatial_split_by_x(points: np.ndarray, labels: np.ndarray,
                      train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
    """
    Split data spatially by X coordinate bounds.

    Args:
        points: Normalized points (N, 3)
        labels: Labels (N,)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (test = 1 - train - val)

    Returns:
        train_points, train_labels, val_points, val_labels, test_points, test_labels
    """
    # Sort by X coordinate
    x_coords = points[:, 0]
    sorted_idx = np.argsort(x_coords)

    sorted_points = points[sorted_idx]
    sorted_labels = labels[sorted_idx]

    n_total = points.shape[0]
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split indices
    train_idx = sorted_idx[:n_train]
    val_idx = sorted_idx[n_train:n_train + n_val]
    test_idx = sorted_idx[n_train + n_val:]

    return (points[train_idx], labels[train_idx],
            points[val_idx], labels[val_idx],
            points[test_idx], labels[test_idx])


def stratified_split(points: np.ndarray, labels: np.ndarray,
                    train_ratio: float = 0.7, val_ratio: float = 0.15,
                    seed: int = 42) -> Tuple:
    """
    Split data with stratification to maintain class proportions.

    Args:
        points: Normalized points (N, 3)
        labels: Labels (N,)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        train_points, train_labels, val_points, val_labels, test_points, test_labels
    """
    rng = np.random.default_rng(seed)

    # Separate by class
    class_0_mask = labels == 0
    class_1_mask = labels == 1

    points_0 = points[class_0_mask]
    labels_0 = labels[class_0_mask]
    points_1 = points[class_1_mask]
    labels_1 = labels[class_1_mask]

    print(f"Class 0 (background): {points_0.shape[0]:,} points")
    print(f"Class 1 (electrical): {points_1.shape[0]:,} points")

    # Split each class separately
    def split_class_data(cls_points, cls_labels):
        n_total = cls_points.shape[0]
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Random permutation
        idx = rng.permutation(n_total)

        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        return (cls_points[train_idx], cls_labels[train_idx],
                cls_points[val_idx], cls_labels[val_idx],
                cls_points[test_idx], cls_labels[test_idx])

    # Split each class
    train_0, labels_train_0, val_0, labels_val_0, test_0, labels_test_0 = split_class_data(points_0, labels_0)
    train_1, labels_train_1, val_1, labels_val_1, test_1, labels_test_1 = split_class_data(points_1, labels_1)

    # Combine
    train_points = np.concatenate([train_0, train_1], axis=0)
    train_labels = np.concatenate([labels_train_0, labels_train_1], axis=0)
    val_points = np.concatenate([val_0, val_1], axis=0)
    val_labels = np.concatenate([labels_val_0, labels_val_1], axis=0)
    test_points = np.concatenate([test_0, test_1], axis=0)
    test_labels = np.concatenate([labels_test_0, labels_test_1], axis=0)

    return train_points, train_labels, val_points, val_labels, test_points, test_labels


def save_split_to_las(points: np.ndarray, labels: np.ndarray, center: np.ndarray,
                     scale: float, output_path: str) -> None:
    """Save split data to LAS file."""
    # Denormalize points
    denormalized = points * scale + center

    # Create LAS file
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="binary_class", type=np.uint8))

    las = laspy.LasData(header)

    # Set coordinates
    las.x = denormalized[:, 0]
    las.y = denormalized[:, 1]
    las.z = denormalized[:, 2]

    # Set classification
    las.classification = labels

    # Save
    las.write(output_path)


def print_split_statistics(name: str, points: np.ndarray, labels: np.ndarray) -> None:
    """Print statistics for a data split."""
    unique, counts = np.unique(labels, return_counts=True)
    total = labels.shape[0]

    print(f"\n{name} Split:")
    print("-" * 20)
    print(f"Total points: {total:,}")

    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        class_name = "Electrical" if cls == 1 else "Background"
        print(f"  Class {cls} ({class_name}): {count:,} ({percentage:.1f}%)")


def main():
    """Main function."""
    # Paths
    input_file = Path(__file__).parent / "data_split" / "binary_merged_all.las"
    output_dir = Path(__file__).parent / "data_split"

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        print("Run prepare_binary_dataset.py first!")
        sys.exit(1)

    # Load data
    print(f"Loading binary dataset: {input_file}")
    points, labels, center, scale = load_binary_las(str(input_file))

    print(f"Loaded {points.shape[0]:,} points")
    print(f"Normalization center: {center}")
    print(f"Normalization scale: {scale:.6f}")

    # Choose split method
    print("\nSplit method options:")
    print("1. Spatial split (by X coordinate) - maintains spatial coherence")
    print("2. Stratified random split - maintains class proportions")

    choice = input("Choose split method (1 or 2, default=1): ").strip()
    if choice == "2":
        split_method = "stratified"
        print("Using stratified random split...")
        splits = stratified_split(points, labels, train_ratio=0.7, val_ratio=0.15)
    else:
        split_method = "spatial"
        print("Using spatial split by X coordinate...")
        splits = spatial_split_by_x(points, labels, train_ratio=0.7, val_ratio=0.15)

    train_points, train_labels, val_points, val_labels, test_points, test_labels = splits

    # Print statistics
    print_split_statistics("Train", train_points, train_labels)
    print_split_statistics("Validation", val_points, val_labels)
    print_split_statistics("Test", test_points, test_labels)

    # Save splits
    print("\nSaving splits...")
    save_split_to_las(train_points, train_labels, center, scale,
                     str(output_dir / "train_binary.las"))
    save_split_to_las(val_points, val_labels, center, scale,
                     str(output_dir / "val_binary.las"))
    save_split_to_las(test_points, test_labels, center, scale,
                     str(output_dir / "test_binary.las"))

    print("\nSplit complete!")
    print(f"Train: {output_dir / 'train_binary.las'}")
    print(f"Val: {output_dir / 'val_binary.las'}")
    print(f"Test: {output_dir / 'test_binary.las'}")

    # Save split metadata
    metadata = {
        "split_method": split_method,
        "total_points": points.shape[0],
        "train_points": train_points.shape[0],
        "val_points": val_points.shape[0],
        "test_points": test_points.shape[0],
        "center": center.tolist(),
        "scale": scale,
        "class_distribution": {
            "train": {int(k): int(v) for k, v in zip(*np.unique(train_labels, return_counts=True))},
            "val": {int(k): int(v) for k, v in zip(*np.unique(val_labels, return_counts=True))},
            "test": {int(k): int(v) for k, v in zip(*np.unique(test_labels, return_counts=True))}
        }
    }

    import json
    metadata_file = output_dir / "binary_split_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    main()