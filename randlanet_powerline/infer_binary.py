#!/usr/bin/env python3
"""
Binary Inference Script for Electric Pole & Line Detection

Runs inference on new LAS files using a trained binary RandLA-Net model.
"""

import argparse
import sys
from pathlib import Path

import laspy
import numpy as np
import torch

# Import config and utilities
from config_binary import *
from utils_binary import *

# Import existing modules
try:
    from data.las_dataset import _read_xyz_and_labels, normalize_points
    from models.randlanet import RandLANet
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from randlanet_powerline.data.las_dataset import _read_xyz_and_labels, normalize_points
    from randlanet_powerline.models.randlanet import RandLANet


def process_las_file_in_chunks(las_path: str, checkpoint_path: str, output_path: str,
                              chunk_size: int = INFERENCE_CHUNK_SIZE,
                              batch_size: int = INFERENCE_BATCH_SIZE) -> None:
    """
    Process a large LAS file in chunks to avoid memory issues.

    Args:
        las_path: Path to input LAS file
        checkpoint_path: Path to model checkpoint
        output_path: Path to save predictions
        chunk_size: Number of points per chunk
        batch_size: Batch size for inference
    """
    device = setup_device()

    # -----------------------------------------------------------------------
    # Load checkpoint and model
    # -----------------------------------------------------------------------
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    center = checkpoint['center']
    scale = checkpoint['scale']
    num_classes = checkpoint.get('num_classes', NUM_CLASSES)

    model = RandLANet(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # -----------------------------------------------------------------------
    # Load input LAS file
    # -----------------------------------------------------------------------
    print(f"Loading input LAS: {las_path}")
    las = laspy.read(las_path)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)

    print(f"Input points: {xyz.shape[0]:,}")

    # -----------------------------------------------------------------------
    # Normalize points
    # -----------------------------------------------------------------------
    normalized_xyz, center, scale = normalize_points(xyz)
    print(f"Normalization: center={center}, scale={scale:.6f}")

    # -----------------------------------------------------------------------
    # Process in chunks
    # -----------------------------------------------------------------------
    all_predictions = []
    all_confidences = []

    num_chunks = int(np.ceil(xyz.shape[0] / chunk_size))
    print(f"Processing in {num_chunks} chunks of {chunk_size} points each...")

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, xyz.shape[0])

        chunk_xyz = normalized_xyz[start_idx:end_idx]
        chunk_size_actual = chunk_xyz.shape[0]

        print(f"Processing chunk {i+1}/{num_chunks} ({chunk_size_actual} points)")

        # Pad chunk to batch_size * num_points if needed
        num_batches = int(np.ceil(chunk_size_actual / NUM_POINTS))
        total_points_needed = num_batches * NUM_POINTS

        if chunk_size_actual < total_points_needed:
            # Pad with zeros
            padding = np.zeros((total_points_needed - chunk_size_actual, 3), dtype=np.float32)
            chunk_xyz_padded = np.concatenate([chunk_xyz, padding], axis=0)
        else:
            chunk_xyz_padded = chunk_xyz

        # Reshape for model input: (num_batches, NUM_POINTS, 3)
        chunk_input = chunk_xyz_padded.reshape(num_batches, NUM_POINTS, 3)

        # Convert to tensor
        chunk_tensor = torch.from_numpy(chunk_input).to(device)

        # Inference
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=USE_AMP and device.type == "cuda"):
                logits = model(chunk_tensor)  # Shape: (num_batches, num_classes, NUM_POINTS)

            # Get predictions and confidences
            preds = torch.argmax(logits, dim=1).cpu().numpy()  # Shape: (num_batches, NUM_POINTS)

            # Confidence scores (probability of class 1)
            softmax_logits = torch.softmax(logits, dim=1).cpu().numpy()
            confs = softmax_logits[:, 1, :]  # Probability of electrical class

        # Flatten and trim padding
        chunk_preds = preds.flatten()[:chunk_size_actual]
        chunk_confs = confs.flatten()[:chunk_size_actual]

        all_predictions.append(chunk_preds)
        all_confidences.append(chunk_confs)

    # Concatenate all chunks
    final_predictions = np.concatenate(all_predictions, axis=0)
    final_confidences = np.concatenate(all_confidences, axis=0)

    print(f"Inference complete! Predictions shape: {final_predictions.shape}")

    # -----------------------------------------------------------------------
    # Save predictions to LAS file
    # -----------------------------------------------------------------------
    save_predictions_to_las(
        normalized_xyz,  # Use normalized coordinates for saving
        final_predictions,
        center, scale,
        output_path,
        final_confidences
    )

    # -----------------------------------------------------------------------
    # Print statistics
    # -----------------------------------------------------------------------
    unique_preds, counts = np.unique(final_predictions, return_counts=True)
    total = len(final_predictions)

    print("\nPrediction Statistics:")
    print("-" * 30)
    for cls, count in zip(unique_preds, counts):
        percentage = (count / total) * 100
        class_name = CLASS_NAMES[int(cls)]
        print(f"Class {cls} ({class_name}): {count:,} points ({percentage:.1f}%)")

    # Confidence statistics
    print("\nConfidence Statistics:")
    print("-" * 30)
    print(f"Mean confidence: {final_confidences.mean():.4f}")
    print(f"Std confidence: {final_confidences.std():.4f}")
    print(f"Min confidence: {final_confidences.min():.4f}")
    print(f"Max confidence: {final_confidences.max():.4f}")

    # High confidence predictions
    high_conf_mask = final_confidences > CONFIDENCE_THRESHOLD
    high_conf_electrical = np.sum((final_predictions == 1) & high_conf_mask)
    print(f"High confidence electrical predictions (> {CONFIDENCE_THRESHOLD}): {high_conf_electrical:,}")


def main():
    parser = argparse.ArgumentParser(description="Run binary inference on LAS files")
    parser.add_argument("--checkpoint", type=str, default=str(BEST_CHECKPOINT),
                        help="Path to model checkpoint")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input LAS file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save predictions (default: auto-generated)")
    parser.add_argument("--chunk_size", type=int, default=INFERENCE_CHUNK_SIZE,
                        help="Number of points to process per chunk")
    parser.add_argument("--batch_size", type=int, default=INFERENCE_BATCH_SIZE,
                        help="Batch size for inference")
    args = parser.parse_args()

    # Validate inputs
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Auto-generate output filename if not provided
    if args.output_file is None:
        input_name = Path(args.input_file).stem
        args.output_file = str(RESULTS_DIR / f"{input_name}_predicted_binary.las")

    # Create output directory if needed
    Path(args.output_file).parent.mkdir(exist_ok=True)

    print("Binary Inference Configuration:")
    print("-" * 40)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 40)

    # Run inference
    try:
        process_las_file_in_chunks(
            args.input_file,
            args.checkpoint,
            args.output_file,
            args.chunk_size,
            args.batch_size
        )
        print(f"\nInference successful! Results saved to: {args.output_file}")
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()