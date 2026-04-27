#!/usr/bin/env python3
"""
Binary Dataset Preparation for Electric Pole & Line Detection

This script converts the 8-class labeled LAS dataset to binary classification:
- Class 0: Background (all non-electrical classes merged)
- Class 1: Electrical poles and lines

Usage:
    python prepare_binary_dataset.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import laspy
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.las_dataset import _read_xyz_and_labels, normalize_points


def load_classified_las_files(dataset_dir: str) -> List[str]:
    """Get all 8 classified LAS files."""
    classified_files = [
        "Classified_Bill_Board.las",
        "Classified_Building.las",
        "Classified_Electrical_Pole & Line.las",
        "Classified_Ground.las",
        "Classified_High-Vegetation.las",
        "Classified_Low-Vegetation.las",
        "Classified_Road.las",
        "Classified_Wall.las"
    ]

    files = []
    for fname in classified_files:
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required LAS file not found: {fpath}")
        files.append(fpath)

    return files


def convert_to_binary_labels(labels: np.ndarray, is_electrical: bool) -> np.ndarray:
    """Convert multi-class labels to binary."""
    if is_electrical:
        # Electrical poles & lines -> class 1
        return np.ones_like(labels, dtype=np.uint8)
    else:
        # All other classes -> class 0
        return np.zeros_like(labels, dtype=np.uint8)


def merge_las_files_to_binary(files: List[str], output_path: str, max_points: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load all LAS files, convert to binary labels, merge, and normalize.

    Args:
        files: List of LAS file paths
        output_path: Path to save merged binary LAS file
        max_points: Maximum total points (random sampling if exceeded)

    Returns:
        points: Normalized points (N, 3)
        binary_labels: Binary labels (N,)
        center: Center point for denormalization
        scale: Scale factor for denormalization
    """
    all_points = []
    all_labels = []

    print("Loading and converting LAS files to binary...")
    for fpath in tqdm(files):
        fname = os.path.basename(fpath)
        is_electrical = "Electrical_Pole" in fname

        # Load points and original labels
        xyz, _ = _read_xyz_and_labels(fpath)

        # Convert to binary labels
        binary_labels = convert_to_binary_labels(np.zeros(xyz.shape[0]), is_electrical)

        all_points.append(xyz)
        all_labels.append(binary_labels)

        print(f"  {fname}: {xyz.shape[0]:,} points, class {1 if is_electrical else 0}")

    # Concatenate all data
    points = np.concatenate(all_points, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"\nTotal points before processing: {points.shape[0]:,}")

    # Random sampling if needed
    if max_points is not None and points.shape[0] > max_points:
        print(f"Random sampling to {max_points:,} points...")
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]
        labels = labels[idx]

    # Normalize points
    print("Normalizing points...")
    normalized_points, center, scale = normalize_points(points)

    # Save merged binary LAS file
    print(f"Saving merged binary LAS to: {output_path}")
    save_binary_las(output_path, normalized_points, labels, center, scale)

    return normalized_points, labels, center, scale


def save_binary_las(output_path: str, points: np.ndarray, labels: np.ndarray,
                   center: np.ndarray, scale: float) -> None:
    """Save points and binary labels to LAS file."""
    # Create LAS file
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="binary_class", type=np.uint8))

    las = laspy.LasData(header)

    # Set coordinates (denormalize for saving)
    denormalized = points * scale + center
    las.x = denormalized[:, 0]
    las.y = denormalized[:, 1]
    las.z = denormalized[:, 2]

    # Set classification
    las.classification = labels

    # Save
    las.write(output_path)


def print_class_distribution(labels: np.ndarray) -> None:
    """Print binary class distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    total = labels.shape[0]

    print("\nBinary Class Distribution:")
    print("-" * 30)
    for cls, count in zip(unique, counts):
        percentage = (count / total) * 100
        class_name = "Electrical_Poles_Lines" if cls == 1 else "Background"
        print(f"Class {cls} ({class_name}): {count:,} points ({percentage:.1f}%)")


def main():
    """Main function."""
    # Paths
    dataset_dir = Path(__file__).parent.parent / "Dataset"
    output_dir = Path(__file__).parent / "data_split"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "binary_merged_all.las"

    # Check if output already exists
    if output_file.exists():
        print(f"Output file already exists: {output_file}")
        overwrite = input("Overwrite? (y/N): ").lower().strip()
        if overwrite != 'y':
            print("Exiting.")
            return

    # Load and process files
    try:
        files = load_classified_las_files(str(dataset_dir))
        points, labels, center, scale = merge_las_files_to_binary(files, str(output_file))

        # Print statistics
        print_class_distribution(labels)
        print(f"\nNormalization center: {center}")
        print(f"Normalization scale: {scale:.6f}")
        print(f"\nBinary dataset saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()