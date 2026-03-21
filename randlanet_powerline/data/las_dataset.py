import os
from glob import glob
from typing import List, Optional, Sequence, Tuple

import laspy
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_POSITIVE_PATTERNS = ["electrical_pole", "line"]


def discover_las_files(dataset_dir: str) -> List[str]:
    pattern = os.path.join(dataset_dir, "*.las")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No LAS files found in: {dataset_dir}")
    return files


def is_positive_file(path: str, positive_patterns: Sequence[str]) -> bool:
    name = os.path.basename(path).lower()
    return any(pattern.lower() in name for pattern in positive_patterns)


def _read_xyz(path: str) -> np.ndarray:
    las = laspy.read(path)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    if xyz.shape[0] == 0:
        raise ValueError(f"No points found in LAS file: {path}")
    return xyz


def load_points_and_labels(
    dataset_dir: str,
    positive_patterns: Optional[Sequence[str]] = None,
    max_points_per_file: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    patterns = positive_patterns or DEFAULT_POSITIVE_PATTERNS
    point_chunks = []
    label_chunks = []

    for path in discover_las_files(dataset_dir):
        xyz = _read_xyz(path)

        if max_points_per_file is not None and xyz.shape[0] > max_points_per_file:
            idx = np.random.choice(xyz.shape[0], max_points_per_file, replace=False)
            xyz = xyz[idx]

        label = 1 if is_positive_file(path, patterns) else 0
        labels = np.full((xyz.shape[0],), label, dtype=np.int64)

        point_chunks.append(xyz)
        label_chunks.append(labels)

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


def compute_class_weights(labels: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    inv = 1.0 / (counts + epsilon)
    weights = inv / inv.sum() * 2.0
    return weights.astype(np.float32)
