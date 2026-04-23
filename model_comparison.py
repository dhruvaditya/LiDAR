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


def run_inference(model, test_loader, device, xyz):
    """Run inference and return predictions"""
    preds = np.zeros((xyz.shape[0],), dtype=np.int64)

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

    return preds


def plot_confusion_matrix(cm: torch.Tensor, num_classes: int, title: str, output_path: str = None):
    """Plot confusion matrix as a heatmap."""
    cm_np = cm.numpy() if isinstance(cm, torch.Tensor) else cm

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_np, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {title}', fontsize=14, fontweight='bold')
    ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax.set_yticklabels([f'Class {i}' for i in range(num_classes)])

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")
    plt.show()


def plot_per_class_metrics(metrics: dict, num_classes: int, title: str, output_path: str = None):
    """Plot per-class metrics (IoU, F1, Precision, Recall)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Per-Class Performance Metrics - {title}', fontsize=16, fontweight='bold')

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


def plot_model_comparison(metrics_best: dict, metrics_last: dict, num_classes: int, output_path: str = None):
    """Plot comparison between best and last model metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison: Best vs Last Checkpoint', fontsize=16, fontweight='bold')

    class_labels = [f'Class {i}' for i in range(num_classes)]
    x_pos = np.arange(num_classes)
    width = 0.35

    metrics = [
        ('iou_per_class', 'IoU', 'skyblue', 'lightblue'),
        ('f1_per_class', 'F1 Score', 'lightgreen', 'palegreen'),
        ('precision_per_class', 'Precision', 'salmon', 'lightsalmon'),
        ('recall_per_class', 'Recall', 'plum', 'thistle')
    ]

    for idx, (metric_key, title, color1, color2) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        best_values = metrics_best[metric_key]
        last_values = metrics_last[metric_key]

        bars1 = ax.bar(x_pos - width/2, best_values, width, label='Best Model', color=color1, edgecolor='navy')
        bars2 = ax.bar(x_pos + width/2, last_values, width, label='Last Model', color=color2, edgecolor='darkred')

        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(class_labels)
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars1, best_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        for bar, val in zip(bars2, last_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to: {output_path}")
    plt.show()


def plot_overall_metrics_comparison(metrics_best: dict, metrics_last: dict, output_path: str = None):
    """Plot overall metrics comparison between models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_names = ['Accuracy', 'Mean IoU', 'Mean F1']
    best_values = [metrics_best['accuracy'], metrics_best['iou'], metrics_best['mean_f1']]
    last_values = [metrics_last['accuracy'], metrics_last['iou'], metrics_last['mean_f1']]

    x_pos = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, best_values, width, label='Best Model', color='steelblue', edgecolor='navy')
    bars2 = ax.bar(x_pos + width/2, last_values, width, label='Last Model', color='lightcoral', edgecolor='darkred')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Metrics Comparison: Best vs Last Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, best_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    for bar, val in zip(bars2, last_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Overall metrics comparison saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare best.pt and last.pt models with comprehensive statistical plots")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Root data folder or data/data_split")
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    args = parser.parse_args()

    args.dataset_dir = resolve_dataset_dir(args.dataset_dir)
    print(f"Using dataset_dir: {args.dataset_dir}")

    # Load test data
    test_file = os.path.join(args.dataset_dir, "test_randla_net.las")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    las = laspy.read(test_file)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    labels = np.array(las.classification).astype(np.int64)

    # Find checkpoints
    best_checkpoint = resolve_checkpoint("randlanet_powerline/checkpoints/best.pt")
    last_checkpoint = resolve_checkpoint("randlanet_powerline/checkpoints/last.pt")

    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Last checkpoint: {last_checkpoint}")

    # Load checkpoints
    ckpt_best = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    ckpt_last = torch.load(last_checkpoint, map_location="cpu", weights_only=False)

    center = ckpt_best.get("center", np.zeros(3, dtype=np.float32))
    scale = float(ckpt_best.get("scale", 1.0))
    num_classes = int(ckpt_best.get("num_classes", 3))

    print(f"Center: {center}, Scale: {scale}, Num classes: {num_classes}")

    # Normalize points
    xyz_norm = (xyz - center) / max(scale, 1e-8)

    # Create test dataset and dataloader
    test_dataset = TestDataset(xyz_norm, args.chunk_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and evaluate best model
    print("\n" + "="*50)
    print("EVALUATING BEST MODEL")
    print("="*50)

    model_best = RandLANet(num_classes=num_classes)
    model_best.load_state_dict(ckpt_best["model_state_dict"], strict=False)
    model_best.to(device)
    model_best.eval()

    preds_best = run_inference(model_best, test_loader, device, xyz)
    pred_tensor_best = torch.from_numpy(preds_best.astype(np.int64))
    label_tensor = torch.from_numpy(labels.astype(np.int64))
    cm_best = confusion_multiclass(pred_tensor_best, label_tensor, num_classes)
    metrics_best = multiclass_metrics(cm_best)

    print(f"Best Model - Total points: {len(xyz)}")
    print(f"Best Model - Accuracy: {metrics_best['accuracy']:.4f}")
    print(f"Best Model - Mean IoU: {metrics_best['iou']:.4f}")
    print(f"Best Model - Mean F1: {metrics_best['mean_f1']:.4f}")

    # Load and evaluate last model
    print("\n" + "="*50)
    print("EVALUATING LAST MODEL")
    print("="*50)

    model_last = RandLANet(num_classes=num_classes)
    model_last.load_state_dict(ckpt_last["model_state_dict"], strict=False)
    model_last.to(device)
    model_last.eval()

    preds_last = run_inference(model_last, test_loader, device, xyz)
    pred_tensor_last = torch.from_numpy(preds_last.astype(np.int64))
    cm_last = confusion_multiclass(pred_tensor_last, label_tensor, num_classes)
    metrics_last = multiclass_metrics(cm_last)

    print(f"Last Model - Total points: {len(xyz)}")
    print(f"Last Model - Accuracy: {metrics_last['accuracy']:.4f}")
    print(f"Last Model - Mean IoU: {metrics_last['iou']:.4f}")
    print(f"Last Model - Mean F1: {metrics_last['mean_f1']:.4f}")

    # Create output directory
    output_dir = "model_comparison_results"
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("\n" + "="*50)
    print("GENERATING COMPARISON PLOTS")
    print("="*50)

    # Confusion matrices
    plot_confusion_matrix(cm_best, num_classes, "Best Model",
                         output_path=f"{output_dir}/confusion_matrix_best.png")
    plot_confusion_matrix(cm_last, num_classes, "Last Model",
                         output_path=f"{output_dir}/confusion_matrix_last.png")

    # Per-class metrics
    plot_per_class_metrics(metrics_best, num_classes, "Best Model",
                          output_path=f"{output_dir}/per_class_metrics_best.png")
    plot_per_class_metrics(metrics_last, num_classes, "Last Model",
                          output_path=f"{output_dir}/per_class_metrics_last.png")

    # Comparison plots
    plot_model_comparison(metrics_best, metrics_last, num_classes,
                         output_path=f"{output_dir}/model_comparison.png")
    plot_overall_metrics_comparison(metrics_best, metrics_last,
                                   output_path=f"{output_dir}/overall_metrics_comparison.png")

    print(f"\nAll plots saved to: {output_dir}/")
    print("Generated files:")
    print("- confusion_matrix_best.png")
    print("- confusion_matrix_last.png")
    print("- per_class_metrics_best.png")
    print("- per_class_metrics_last.png")
    print("- model_comparison.png")
    print("- overall_metrics_comparison.png")


if __name__ == "__main__":
    main()