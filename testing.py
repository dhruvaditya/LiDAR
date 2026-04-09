import argparse
import os
import sys

import laspy
import numpy as np
import torch

try:
    from randlanet_powerline.data.las_dataset import load_points_and_labels, normalize_points
    from randlanet_powerline.models.randlanet import RandLANet
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from randlanet_powerline.data.las_dataset import load_points_and_labels, normalize_points
    from randlanet_powerline.models.randlanet import RandLANet


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

    for candidate in ["checkpoints/best.pt", "checkpoints/last.pt"]:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "No checkpoint found. Provide --checkpoint or place best.pt/last.pt in checkpoints/."
    )


def chunk_indices(total: int, chunk_size: int):
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


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
    return {
        "accuracy": (tp.sum() / cm.sum().clamp(min=1e-8)).item(),
        "iou": iou_per_class.mean().item(),
        "mean_f1": f1_per_class.mean().item(),
        "iou_per_class": iou_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
    }


def load_test_points(dataset_dir: str, filename: str = "test_randla_net.las"):
    file_path = os.path.join(dataset_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test LAS not found: {file_path}")

    xyz, labels = load_points_and_labels(dataset_dir, file_names=[filename])
    return xyz, labels


def save_predictions(las, preds, output_path: str):
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
    model.load_state_dict(ckpt["model_state_dict"])
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
    cm_total = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for start, end in chunk_indices(xyz_norm.shape[0], args.chunk_size):
            chunk = xyz_norm[start:end]
            if chunk.shape[0] < args.chunk_size:
                pad = args.chunk_size - chunk.shape[0]
                pad_idx = np.random.choice(chunk.shape[0], pad, replace=True)
                chunk = np.concatenate([chunk, chunk[pad_idx]], axis=0)

            inp = torch.from_numpy(chunk).unsqueeze(0).to(device)
            logits = model(inp)
            out = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            preds[start:end] = out[: end - start]

    pred_tensor = torch.from_numpy(preds.astype(np.int64))
    label_tensor = torch.from_numpy(labels.astype(np.int64))
    cm_total += confusion_multiclass(pred_tensor, label_tensor, num_classes)
    metrics = multiclass_metrics(cm_total)

    print("\n========== TEST RESULTS ==========")
    print(f"Total points: {len(xyz)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean IoU: {metrics['iou']:.4f}")
    print(f"Mean F1: {metrics['mean_f1']:.4f}")
    for idx, val in enumerate(metrics['iou_per_class']):
        print(f"Class {idx} IoU: {val:.4f}")
    for idx, val in enumerate(metrics['f1_per_class']):
        print(f"Class {idx} F1:  {val:.4f}")

    print(f"\nWriting predictions to: {args.output_las}")
    save_predictions(las, preds, args.output_las)


if __name__ == "__main__":
    main()
