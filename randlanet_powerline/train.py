import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from randlanet_powerline.data.las_dataset import (
    RandomPointBlockDataset,
    compute_class_weights,
    load_points_and_labels,
    normalize_points,
    train_val_split,
)
from randlanet_powerline.models.randlanet import RandLANet
from randlanet_powerline.utils.metrics import binary_metrics, confusion_binary


def validate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_tp = total_tn = total_fp = total_fn = 0

    with torch.no_grad():
        for xyz, y in loader:
            xyz = xyz.to(device)
            y = y.to(device)

            logits = model(xyz)
            loss = criterion(logits, y)
            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            tp, tn, fp, fn = confusion_binary(pred, y)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

    metrics = binary_metrics(total_tp, total_tn, total_fp, total_fn)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train RandLA-Net for power-line detection")
    parser.add_argument("--dataset_dir", type=str, default="Dataset")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--train_steps", type=int, default=250)
    parser.add_argument("--val_steps", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_points_per_file", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    max_pts = args.max_points_per_file if args.max_points_per_file > 0 else None
    
    # Load training data
    print("Loading training data from train_randla_net.las...")
    try:
        x_train, y_train = load_points_and_labels(args.dataset_dir, 
                                                 file_names=["train_randla_net.las"], 
                                                 max_points_per_file=max_pts)
        print(f"Training data: {x_train.shape[0]} points")
    except FileNotFoundError:
        raise FileNotFoundError(f"train_randla_net.las not found in {args.dataset_dir}")
    
    # Normalize training data
    x_train, center, scale = normalize_points(x_train)
    
    # Load validation data if available
    val_data_path = os.path.join(args.dataset_dir, "val_randla_net.las")
    if os.path.exists(val_data_path):
        print("Loading validation data from val_randla_net.las...")
        x_val, y_val = load_points_and_labels(args.dataset_dir, 
                                             file_names=["val_randla_net.las"], 
                                             max_points_per_file=max_pts)
        # Apply same normalization as training data
        x_val = (x_val - center) / max(scale, 1e-8)
        print(f"Validation data: {x_val.shape[0]} points")
    else:
        print("No validation data found, using portion of training data for validation...")
        # Fallback to splitting training data
        x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, 
                                                        val_ratio=args.val_ratio, 
                                                        seed=args.seed)

    train_ds = RandomPointBlockDataset(
        x_train,
        y_train,
        num_points=args.num_points,
        steps_per_epoch=args.train_steps,
        augment=True,
    )
    val_ds = RandomPointBlockDataset(
        x_val,
        y_val,
        num_points=args.num_points,
        steps_per_epoch=args.val_steps,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RandLANet(num_classes=3).to(device)

    class_weights = compute_class_weights(y_train, num_classes=3)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for xyz, y in train_loader:
            xyz = xyz.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xyz)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = validate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "center": center,
            "scale": scale,
            "args": vars(args),
        }

        torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))

    print(f"Training complete. Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
