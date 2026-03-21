import argparse

import numpy as np
import torch

from randlanet_powerline.data.las_dataset import (
    RandomPointBlockDataset,
    load_points_and_labels,
    normalize_points,
    train_val_split,
)
from randlanet_powerline.models.randlanet import RandLANet
from randlanet_powerline.utils.metrics import binary_metrics, confusion_binary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RandLA-Net checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="Dataset")
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--val_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = RandLANet(num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    points, labels = load_points_and_labels(args.dataset_dir)
    points, _, _ = normalize_points(points)
    _, _, x_val, y_val = train_val_split(points, labels, val_ratio=args.val_ratio, seed=args.seed)

    val_ds = RandomPointBlockDataset(
        x_val,
        y_val,
        num_points=args.num_points,
        steps_per_epoch=args.val_steps,
        augment=False,
    )

    loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    total_tp = total_tn = total_fp = total_fn = 0
    with torch.no_grad():
        for xyz, y in loader:
            xyz = xyz.to(device)
            y = y.to(device)

            logits = model(xyz)
            pred = torch.argmax(logits, dim=1)
            tp, tn, fp, fn = confusion_binary(pred, y)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

    metrics = binary_metrics(total_tp, total_tn, total_fp, total_fn)
    print("Evaluation Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
