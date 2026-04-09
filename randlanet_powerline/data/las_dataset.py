import os
from glob import glob
from typing import List, Optional, Sequence, Tuple

import laspy
import numpy as np
import torch
from torch.utils.data import Dataset


POWERLINE_PATTERNS = ["line", "power"]
PYLON_PATTERNS = ["pylon", "pole", "electrical_pole", "tower"]


def discover_las_files(dataset_dir: str) -> List[str]:
    pattern = os.path.join(dataset_dir, "*.las")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No LAS files found in: {dataset_dir}")
    return files


def get_file_class(path: str, powerline_patterns: Sequence[str] = None, pylon_patterns: Sequence[str] = None) -> int:
    """Classify LAS file into class 0 (background), 1 (power line), or 2 (pylon)."""
    powerline_patterns = powerline_patterns or POWERLINE_PATTERNS
    pylon_patterns = pylon_patterns or PYLON_PATTERNS
    
    name = os.path.basename(path).lower()
    
    # Check for pylon/pole first (class 2)
    if any(pattern.lower() in name for pattern in pylon_patterns):
        return 2
    # Then check for power line (class 1)
    elif any(pattern.lower() in name for pattern in powerline_patterns):
        return 1
    # Otherwise background (class 0)
    else:
        return 0


def _read_xyz_and_labels(path: str) -> Tuple[np.ndarray, np.ndarray]:
    las = laspy.read(path)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    labels = np.array(las.classification).astype(np.int64)
    if xyz.shape[0] == 0:
        raise ValueError(f"No points found in LAS file: {path}")
    return xyz, labels


def load_points_and_labels(
    dataset_dir: str,
    file_names: Optional[List[str]] = None,
    max_points_per_file: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load points and labels from LAS files.
    
    Args:
        dataset_dir: Directory containing LAS files
        file_names: List of specific file names to load (e.g., ['train_randla_net.las'])
                   If None, loads all .las files
        max_points_per_file: Maximum points to load per file (random sampling if exceeded)
    
    Returns:
        points: Combined point cloud (N, 3)
        labels: Combined labels (N,)
    """
    if file_names is None:
        # Load all LAS files
        files = discover_las_files(dataset_dir)
    else:
        # Load specific files
        files = [os.path.join(dataset_dir, fname) for fname in file_names]
        # Check that files exist
        for f in files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"LAS file not found: {f}")
    
    point_chunks = []
    label_chunks = []
    
    for path in files:
        xyz, labels = _read_xyz_and_labels(path)
        
        if max_points_per_file is not None and xyz.shape[0] > max_points_per_file:
            idx = np.random.choice(xyz.shape[0], max_points_per_file, replace=False)
            xyz = xyz[idx]
            labels = labels[idx]
        
        point_chunks.append(xyz)
        label_chunks.append(labels)
    
    if not point_chunks:
        raise ValueError(f"No data loaded from {dataset_dir}")
    
    points = np.concatenate(point_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    return points, labels


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0, keepdims=True)
    shifted = points - center
    scale = float(np.linalg.norm(shifted, axis=1).max())
    if scale <= 0:
        scale = 1.0
    normalized = shifted / scale
    return normalized.astype(np.float32), center.squeeze(0).astype(np.float32), scale


def train_val_split(
    points: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(points.shape[0])
    rng.shuffle(idx)

    split_idx = int(points.shape[0] * (1.0 - val_ratio))
    train_idx = idx[:split_idx]
    val_idx = idx[split_idx:]

    return points[train_idx], labels[train_idx], points[val_idx], labels[val_idx]


class RandomPointBlockDataset(Dataset):
    """Random sampling dataset - samples random points from the point cloud."""
    def __init__(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        num_points: int,
        steps_per_epoch: int,
        augment: bool = False,
    ) -> None:
        if points.shape[0] != labels.shape[0]:
            raise ValueError("points and labels must have same length")
        self.points = points.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.num_points = int(num_points)
        self.steps_per_epoch = int(steps_per_epoch)
        self.augment = augment

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, index: int):
        _ = index
        total = self.points.shape[0]
        replace = total < self.num_points
        sample_idx = np.random.choice(total, self.num_points, replace=replace)

        xyz = self.points[sample_idx].copy()
        y = self.labels[sample_idx].copy()

        if self.augment:
            xyz = self._augment_xyz(xyz)

        return torch.from_numpy(xyz), torch.from_numpy(y)

    @staticmethod
    def _augment_xyz(xyz: np.ndarray) -> np.ndarray:
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz = xyz @ rot.T

        jitter = np.random.normal(0, 0.005, size=xyz.shape).astype(np.float32)
        xyz = xyz + jitter

        scale = np.random.uniform(0.95, 1.05)
        xyz = xyz * scale

        return xyz.astype(np.float32)


class SpatiallyRegularDataset(Dataset):
    def __init__(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        num_points: int,
        steps_per_epoch: int,
        augment: bool = False,
        noise_init: float = 3.5,
    ) -> None:
        if points.shape[0] != labels.shape[0]:
            raise ValueError("points and labels must have same length")
        self.points = points.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.num_points = int(num_points)
        self.steps_per_epoch = int(steps_per_epoch)
        self.augment = augment
        self.noise_init = noise_init

        # Initialize possibility scores (lower = more likely to be sampled)
        self.possibility = np.random.rand(self.points.shape[0]) * 1e-3
        self.min_possibility = float(np.min(self.possibility))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, index: int):
        _ = index

        # Choose point with minimum possibility (least sampled)
        point_ind = np.argmin(self.possibility)

        # Center point of input region
        center_point = self.points[point_ind:point_ind+1]

        # Add noise to center point
        noise = np.random.normal(scale=self.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)

        # Query nearest neighbors
        if len(self.points) < self.num_points:
            # If we have fewer points than needed, take all
            queried_idx = np.arange(len(self.points))
        else:
            # Compute distances to pick_point
            dists = np.sum(np.square(self.points - pick_point), axis=1)
            # Get indices of closest points
            queried_idx = np.argsort(dists)[:self.num_points]

        # Shuffle indices
        np.random.shuffle(queried_idx)

        # Get points and labels
        xyz = self.points[queried_idx].copy() - pick_point
        y = self.labels[queried_idx].copy()

        # Update possibility scores
        dists = np.sum(np.square((self.points[queried_idx] - pick_point).astype(np.float32)), axis=1)
        delta = np.square(1 - dists / np.max(dists + 1e-6))
        self.possibility[queried_idx] += delta
        self.min_possibility = float(np.min(self.possibility))

        if self.augment:
            xyz = self._augment_xyz(xyz)

        return torch.from_numpy(xyz), torch.from_numpy(y)

    @staticmethod
    def _augment_xyz(xyz: np.ndarray) -> np.ndarray:
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz = xyz @ rot.T

        jitter = np.random.normal(0, 0.005, size=xyz.shape).astype(np.float32)
        xyz = xyz + jitter

        scale = np.random.uniform(0.95, 1.05)
        xyz = xyz * scale

        return xyz.astype(np.float32)


def compute_class_weights(labels: np.ndarray, num_classes: int = 3, epsilon: float = 1e-6) -> np.ndarray:
    """Compute class weights for imbalanced dataset (3 classes by default)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    inv = 1.0 / (counts + epsilon)
    weights = inv / inv.sum() * num_classes
    return weights.astype(np.float32)
