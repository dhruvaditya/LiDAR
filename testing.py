import argparse
import os
import sys

import laspy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, classification_report

try:
    from randlanet_powerline.data.las_dataset import load_points_and_labels, normalize_points
    from randlanet_powerline.models.randlanet import RandLANet
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from randlanet_powerline.data.las_dataset import load_points_and_labels, normalize_points
    from randlanet_powerline.models.randlanet import RandLANet


class TestDataset(Dataset):
    def __init__(self, points: np.ndarray, chunk_size: int):
        self.points = points
        self.chunk_size = chunk_size
        self.indices = []

        for start in range(0, len(points), chunk_size):
            end = min(start + chunk_size, len(points))
            self.indices.append((start, end))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, end = self.indices[idx]
        chunk = self.points[start:end]
        if len(chunk) < self.chunk_size:
            # Pad the chunk to chunk_size
            pad_size = self.chunk_size - len(chunk)
            pad = np.random.choice(len(chunk), pad_size, replace=True)
            chunk = np.concatenate([chunk, chunk[pad]], axis=0)
        return torch.from_numpy(chunk), start, end


def resolve_dataset_dir(dataset_dir: str) -> str:
    # Prefer explicit dataset_dir when valid.
    direct_path = os.path.join(dataset_dir, "test_randla_net.las")
    if os.path.exists(direct_path):
        return dataset_dir

    candidate = os.path.join(dataset_dir, "data_split")
    if os.path.exists(os.path.join(candidate, "test_randla_net.las")):
        return candidate

    # Fallback to the randlanet_powerline package data folder.
    script_root = os.path.dirname(os.path.abspath(__file__))
    panel = os.path.join(script_root, "randlanet_powerline", "data")
    if os.path.exists(os.path.join(panel, "test_randla_net.las")):
        return panel
    panel_split = os.path.join(panel, "data_split")
    if os.path.exists(os.path.join(panel_split, "test_randla_net.las")):
        return panel_split

    # Additional fallback for a top-level data/data_split layout.
    alt = os.path.join(script_root, "data")
    if os.path.exists(os.path.join(alt, "test_randla_net.las")):
        return alt
    alt_split = os.path.join(alt, "data_split")
    if os.path.exists(os.path.join(alt_split, "test_randla_net.las")):
        return alt_split

    return dataset_dir


def resolve_checkpoint(checkpoint: str) -> str:
    if checkpoint and os.path.exists(checkpoint):
        return checkpoint

    script_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "randlanet_powerline/checkpoints/best.pt",
        "randlanet_powerline/checkpoints/last.pt",
        "check_test/best.pt",
        "check_test/last.pt",
        os.path.join(script_root, "checkpoints", "best.pt"),
        os.path.join(script_root, "checkpoints", "last.pt"),
        os.path.join(script_root, "randlanet_powerline", "checkpoints", "best.pt"),
        os.path.join(script_root, "randlanet_powerline", "checkpoints", "last.pt"),
        os.path.join(script_root, "check_test", "best.pt"),
        os.path.join(script_root, "check_test", "last.pt"),
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate):
            print(f"Found checkpoint: {candidate}")
            return candidate

    raise FileNotFoundError(
        "No checkpoint found. Searched in: checkpoints/, randlanet_powerline/checkpoints/, check_test/. "
        "Provide --checkpoint or place best.pt/last.pt in one of these directories."
    )


def confusion_multiclass(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    mask = (target >= 0) & (target < num_classes)
    combined = num_classes * target[mask] + pred[mask]
    cm = torch.bincount(combined, minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)


def multiclass_metrics(cm: torch.Tensor) -> dict:
    cm = cm.float()
    tp = cm.diag()
    fn = cm.sum(dim=1) - tp
    fp = cm.sum(dim=0) - tp
    iou_per_class = tp / (tp + fp + fn).clamp(min=1e-8)
    f1_per_class = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-8)
    precision_per_class = tp / (tp + fp).clamp(min=1e-8)
    recall_per_class = tp / (tp + fn).clamp(min=1e-8)
    return {
        "accuracy": (tp.sum() / cm.sum().clamp(min=1e-8)).item(),
        "iou": iou_per_class.mean().item(),
        "mean_f1": f1_per_class.mean().item(),
        "iou_per_class": iou_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "tp": tp.tolist(),
        "fp": fp.tolist(),
        "fn": fn.tolist(),
    }


def plot_confusion_matrix(cm: torch.Tensor, num_classes: int, output_path: str = None):
    """Plot confusion matrix as a heatmap."""
    cm_np = cm.numpy() if isinstance(cm, torch.Tensor) else cm
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_np, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax.set_yticklabels([f'Class {i}' for i in range(num_classes)])
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")
    plt.show()


def plot_per_class_metrics(metrics: dict, num_classes: int, output_path: str = None):
    """Plot per-class metrics (IoU, F1, Precision, Recall)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    
    class_labels = [f'Class {i}' for i in range(num_classes)]
    x_pos = np.arange(num_classes)
    
    # IoU per class
    ax = axes[0, 0]
    iou_values = metrics['iou_per_class']
    bars = ax.bar(x_pos, iou_values, color='skyblue', edgecolor='navy')
    ax.set_ylabel('IoU', fontsize=11)
    ax.set_title('Intersection over Union (IoU)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, iou_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # F1 per class
    ax = axes[0, 1]
    f1_values = metrics['f1_per_class']
    bars = ax.bar(x_pos, f1_values, color='lightgreen', edgecolor='darkgreen')
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('F1 Score', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, f1_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Precision per class
    ax = axes[1, 0]
    precision_values = metrics['precision_per_class']
    bars = ax.bar(x_pos, precision_values, color='salmon', edgecolor='darkred')
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, precision_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Recall per class
    ax = axes[1, 1]
    recall_values = metrics['recall_per_class']
    bars = ax.bar(x_pos, recall_values, color='plum', edgecolor='purple')
    ax.set_ylabel('Recall', fontsize=11)
    ax.set_title('Recall', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_labels)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, recall_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to: {output_path}")
    plt.show()


def plot_error_analysis(metrics: dict, num_classes: int, output_path: str = None):
    """Plot error analysis (TP, FP, FN)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(num_classes)
    width = 0.25
    
    tp_values = metrics['tp']
    fp_values = metrics['fp']
    fn_values = metrics['fn']
    
    bars1 = ax.bar(x - width, tp_values, width, label='True Positives', color='green', alpha=0.8)
    bars2 = ax.bar(x, fp_values, width, label='False Positives', color='red', alpha=0.8)
    bars3 = ax.bar(x + width, fn_values, width, label='False Negatives', color='orange', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Error Analysis: TP, FP, FN per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Error analysis plot saved to: {output_path}")
    plt.show()


def save_predictions(las, preds, output_path: str, class_filter: int = None):
    if class_filter is not None:
        mask = preds == class_filter
        if not mask.any():
            print(f"No points found for class {class_filter}. Skipping save.")
            return
        # Filter the LAS data
        filtered_las = laspy.LasData(las.header)
        filtered_las.x = las.x[mask]
        filtered_las.y = las.y[mask]
        filtered_las.z = las.z[mask]
        for dim_name in las.point_format.dimension_names:
            if dim_name in {"X", "Y", "Z"}:
                continue
            filtered_las[dim_name] = las[dim_name][mask]
        filtered_las.classification = preds[mask].astype(np.uint8)
        out_las = filtered_las
    else:
        out_las = laspy.LasData(las.header)
        out_las.x = las.x
        out_las.y = las.y
        out_las.z = las.z
        for dim_name in las.point_format.dimension_names:
            if dim_name in {"X", "Y", "Z"}:
                continue
            out_las[dim_name] = las[dim_name]
        out_las.classification = preds.astype(np.uint8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_las.write(output_path)


def main():
    parser = argparse.ArgumentParser(description="Test the last trained RandLA-Net model on test_randla_net.las")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Root data folder or data/data_split")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint file")
    parser.add_argument("--output_las", type=str, default="data/data_split/test_randla_net_predicted.las")
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--save_powerlines_only", action="store_true", help="Save only power line points (class 1) in the output LAS")
    args = parser.parse_args()

    args.dataset_dir = resolve_dataset_dir(args.dataset_dir)
    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"Using dataset_dir: {args.dataset_dir}")
    print(f"Using checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    center = ckpt.get("center", np.zeros(3, dtype=np.float32))
    scale = float(ckpt.get("scale", 1.0))
    num_classes = int(ckpt.get("num_classes", 3))

    model = RandLANet(num_classes=num_classes)
    try:
        missing_keys, unexpected_keys = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing_keys or unexpected_keys:
            print("Warning: Model state_dict loading with mismatches:")
            if missing_keys:
                print(f"  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys}")
            print("Proceeding with partial loading. Results may be inaccurate.")
    except RuntimeError as e:
        print(f"Warning: Failed to load state_dict due to size mismatches: {e}")
        print("Attempting to load matching parameters manually...")
        # Try to load only the matching keys
        state_dict = ckpt["model_state_dict"]
        model_dict = model.state_dict()
        loaded_keys = []
        for k in model_dict:
            if k in state_dict and state_dict[k].shape == model_dict[k].shape:
                model_dict[k] = state_dict[k]
                loaded_keys.append(k)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(loaded_keys)} matching parameters. Results may be inaccurate.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_file = os.path.join(args.dataset_dir, "test_randla_net.las")
    if not os.path.exists(test_file):
        raise FileNotFoundError(
            f"Test file not found. Expected: {test_file}"
        )

    las = laspy.read(test_file)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    labels = np.array(las.classification).astype(np.int64)

    xyz_norm = (xyz - center) / max(scale, 1e-8)
    preds = np.zeros((xyz.shape[0],), dtype=np.int64)

    # Create test dataset and dataloader
    test_dataset = TestDataset(xyz_norm, args.chunk_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch_xyz, batch_starts, batch_ends in test_loader:
            batch_xyz = batch_xyz.to(device)
            batch_logits = model(batch_xyz)
            batch_preds = torch.argmax(batch_logits, dim=1).cpu().numpy()

            for i in range(len(batch_starts)):
                start = batch_starts[i].item()
                end = batch_ends[i].item()
                chunk_pred = batch_preds[i][:end - start]
                preds[start:end] = chunk_pred

    pred_tensor = torch.from_numpy(preds.astype(np.int64))
    label_tensor = torch.from_numpy(labels.astype(np.int64))
    cm_total = confusion_multiclass(pred_tensor, label_tensor, num_classes)
    metrics = multiclass_metrics(cm_total)

    print("\n========== TEST RESULTS ==========")
    print(f"Total points: {len(xyz)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean IoU: {metrics['iou']:.4f}")
    print(f"Mean F1: {metrics['mean_f1']:.4f}")
    print(f"\nPer-Class Metrics:")
    for idx in range(num_classes):
        print(f"  Class {idx}:")
        print(f"    IoU:       {metrics['iou_per_class'][idx]:.4f}")
        print(f"    F1:        {metrics['f1_per_class'][idx]:.4f}")
        print(f"    Precision: {metrics['precision_per_class'][idx]:.4f}")
        print(f"    Recall:    {metrics['recall_per_class'][idx]:.4f}")
        print(f"    TP: {int(metrics['tp'][idx])}, FP: {int(metrics['fp'][idx])}, FN: {int(metrics['fn'][idx])}")
    
    # Generate plots and other statistical parameters
    print("\nGenerating visualization plots...")
    plot_confusion_matrix(cm_total, num_classes, output_path="test_results/confusion_matrix.png")
    plot_per_class_metrics(metrics, num_classes, output_path="test_results/per_class_metrics.png")
    plot_error_analysis(metrics, num_classes, output_path="test_results/error_analysis.png")

    print(f"\nWriting predictions to: {args.output_las}")
    class_filter = 1 if args.save_powerlines_only else None
    save_predictions(las, preds, args.output_las, class_filter)


if __name__ == "__main__":
    main()
