import argparse
import os
import sys

import numpy as np
import torch

try:
    from randlanet_powerline.data.las_dataset import (
        RandomPointBlockDataset,
        load_points_and_labels,
        normalize_points,
        train_val_split,
    )
    from randlanet_powerline.models.randlanet import RandLANet
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from randlanet_powerline.data.las_dataset import (
        RandomPointBlockDataset,
        load_points_and_labels,
        normalize_points,
        train_val_split,
    )
    from randlanet_powerline.models.randlanet import RandLANet


def resolve_dataset_dir(dataset_dir: str) -> str:
    # Prefer the explicit split folder if available.
    direct = os.path.join(dataset_dir, "test_randla_net.las")
    if os.path.exists(direct):
        return dataset_dir

    candidate = os.path.join(dataset_dir, "data_split")
    if os.path.exists(os.path.join(candidate, "test_randla_net.las")):
        return candidate

    # Fallback if only val file exists
    if os.path.exists(os.path.join(dataset_dir, "val_randla_net.las")):
        return dataset_dir
    if os.path.exists(os.path.join(candidate, "val_randla_net.las")):
        return candidate

    return dataset_dir


def confusion_multiclass(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    assert pred.shape == target.shape
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
    return {
        "iou": iou_per_class.mean().item(),
        "mean_f1": f1_per_class.mean().item(),
        "accuracy": (tp.sum() / cm.sum().clamp(min=1e-8)).item(),
        "iou_per_class": iou_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RandLA-Net checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="data/data_split")
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--val_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.dataset_dir = resolve_dataset_dir(args.dataset_dir)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    center = ckpt.get("center", np.zeros(3, dtype=np.float32))
    scale = float(ckpt.get("scale", 1.0))
    num_classes = int(ckpt.get("num_classes", 3))
    
    model = RandLANet(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load test or validation data
    test_data_path = os.path.join(args.dataset_dir, "test_randla_net.las")
    val_data_path = os.path.join(args.dataset_dir, "val_randla_net.las")
    if os.path.exists(test_data_path):
        print("Loading test data from test_randla_net.las...")
        points, labels = load_points_and_labels(args.dataset_dir, file_names=["test_randla_net.las"])
    elif os.path.exists(val_data_path):
        print("Loading validation data from val_randla_net.las...")
        points, labels = load_points_and_labels(args.dataset_dir, file_names=["val_randla_net.las"])
    else:
        print("No test/val LAS found, loading all data and using a validation split...")
        points, labels = load_points_and_labels(args.dataset_dir)
        _, _, points, labels = train_val_split(points, labels, val_ratio=args.val_ratio, seed=args.seed)

    points = (points - center) / max(scale, 1e-8)

    val_ds = RandomPointBlockDataset(
        points,
        labels,
        num_points=args.num_points,
        steps_per_epoch=args.val_steps,
        augment=False,
    )

    loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    cm_total = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for xyz, y in loader:
            xyz = xyz.to(device)
            y = y.to(device)

            logits = model(xyz)
            pred = torch.argmax(logits, dim=1)
            cm_total += confusion_multiclass(pred.reshape(-1), y.reshape(-1), num_classes)

    metrics = multiclass_metrics(cm_total)
    print("Evaluation Metrics")
    print(f"  num_classes: {num_classes}")
    print(f"  dataset_dir: {args.dataset_dir}")
    print(f"  split file: {os.path.basename(test_data_path) if os.path.exists(test_data_path) else ('val_randla_net.las' if os.path.exists(val_data_path) else 'split from all')}")
    print(f"  overall accuracy: {metrics['accuracy']:.4f}")
    print(f"  mean IoU:         {metrics['iou']:.4f}")
    print(f"  mean F1:          {metrics['mean_f1']:.4f}")
    for idx, value in enumerate(metrics['iou_per_class']):
        print(f"  class {idx} IoU: {value:.4f}")
    for idx, value in enumerate(metrics['f1_per_class']):
        print(f"  class {idx} F1:  {value:.4f}")


if __name__ == "__main__":
    main()
