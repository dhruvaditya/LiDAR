#!/usr/bin/env python3
"""
Configuration for Binary Electric Pole & Line Detection with RandLA-Net
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Project root
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "Dataset"
DATA_SPLIT_DIR = DATA_DIR / "data_split"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Create directories if they don't exist
for dir_path in [DATA_SPLIT_DIR, RESULTS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data files
BINARY_MERGED_FILE = DATA_SPLIT_DIR / "binary_merged_all.las"
TRAIN_FILE = DATA_SPLIT_DIR / "train_binary.las"
VAL_FILE = DATA_SPLIT_DIR / "val_binary.las"
TEST_FILE = DATA_SPLIT_DIR / "test_binary.las"

# Checkpoints
BEST_CHECKPOINT = CHECKPOINTS_DIR / "binary_best.pt"
LAST_CHECKPOINT = CHECKPOINTS_DIR / "binary_last.pt"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Number of classes (binary: 0=background, 1=electrical)
NUM_CLASSES = 2

# Class names
CLASS_NAMES = ["Background", "Electrical_Poles_Lines"]

# Model architecture
NUM_POINTS = 4096  # Points per batch
K_NN = 16  # KNN neighbors for local feature aggregation

# RandLA-Net architecture parameters
D_IN = 3  # Input dimension (XYZ)
D_OUT = [32, 64, 128, 256]  # Encoder channel progression
NUM_NEIGHBORS = [16, 16, 16, 16]  # KNN neighbors per level
DECODER_UNITS = [256, 128, 96, 64]  # Decoder channel progression
DROPOUT = 0.3  # Dropout rate

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Loss function weights (Focal + Dice)
FOCAL_WEIGHT = 0.7
DICE_WEIGHT = 0.3
FOCAL_GAMMA = 2.0

# Optimizer
BETAS = (0.9, 0.999)
EPS = 1e-8

# Learning rate scheduler
WARMUP_EPOCHS = 5
COSINE_ANNEALING_T0 = EPOCHS - WARMUP_EPOCHS

# Early stopping
PATIENCE = 20
MIN_DELTA = 0.001

# Mixed precision training
USE_AMP = True

# Random seed
SEED = 42

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Dataset parameters
STEPS_PER_EPOCH = 500  # Training steps per epoch
VAL_STEPS_PER_EPOCH = 100  # Validation steps per epoch

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_PARAMS = {
    "jitter_sigma": 0.002,
    "scale_range": [0.9, 1.1],
    "rotation_angle": 0.1,  # radians
    "elastic_deformation_prob": 0.3,
    "elastic_sigma": 30.0,
    "elastic_alpha": 1.0
}

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

# Inference parameters
INFERENCE_BATCH_SIZE = 1
INFERENCE_CHUNK_SIZE = 8192  # Points per chunk for large files

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.5

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

# Logging
LOG_INTERVAL = 10  # Log every N training steps
SAVE_INTERVAL = 5  # Save checkpoint every N epochs

# TensorBoard logging
USE_TENSORBOARD = True
TENSORBOARD_DIR = PROJECT_ROOT / "tensorboard_logs" / "binary_training"

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

# GPU settings
DEVICE = "cuda"  # or "cpu"
GPU_ID = 0

# Data loading
NUM_WORKERS = 4  # DataLoader workers

# Memory optimization
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

# Metrics to compute
BINARY_METRICS = ["accuracy", "precision", "recall", "f1", "iou", "mIoU"]

# Confusion matrix
SAVE_CONFUSION_MATRIX = True
CONFUSION_MATRIX_FILE = RESULTS_DIR / "confusion_matrix_binary.png"

# Test results
TEST_RESULTS_FILE = RESULTS_DIR / "test_results_binary.json"
TEST_PREDICTIONS_FILE = RESULTS_DIR / "test_predictions_binary.las"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_class_weights(train_labels):
    """Compute class weights for imbalanced binary classification."""
    import numpy as np
    unique, counts = np.unique(train_labels, return_counts=True)
    total = counts.sum()

    # Inverse frequency weighting
    weights = total / (len(unique) * counts)
    weights = weights / weights.sum() * len(unique)  # Normalize

    # Convert to dict
    class_weights = {int(cls): float(w) for cls, w in zip(unique, weights)}
    return class_weights

def print_config():
    """Print current configuration."""
    print("Binary Classification Configuration:")
    print("=" * 50)
    print(f"Classes: {NUM_CLASSES} ({CLASS_NAMES})")
    print(f"Training epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Points per batch: {NUM_POINTS}")
    print(f"Device: {DEVICE}")
    print(f"Use AMP: {USE_AMP}")
    print(f"Random seed: {SEED}")
    print("=" * 50)