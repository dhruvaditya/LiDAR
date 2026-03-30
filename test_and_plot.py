#!/usr/bin/env python
"""
Test the trained model and plot metrics
"""
import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from randlanet_powerline.data.las_dataset import load_points_and_labels, normalize_points, RandomPointBlockDataset
from randlanet_powerline.models.randlanet import RandLANet
from randlanet_powerline.utils.metrics import confusion_binary, binary_metrics


def main():
    parser = argparse.ArgumentParser(description="Test trained model and plot metrics")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", 
                       help="Path to model checkpoint")
    parser.add_argument("--dataset_dir", type=str, default="randlanet_powerline/data/data_split",
                       help="Dataset directory")
    parser.add_argument("--test_file", type=str, default="test_randla_net.las",
                       help="Test file name (relative to dataset_dir)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=4096)
    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("   Please train the model first using:")
        print("   python -m randlanet_powerline.train --dataset_dir randlanet_powerline/data/data_split")
        return

    print(f"✓ Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    center = ckpt.get("center", np.zeros(3, dtype=np.float32))
    scale = float(ckpt.get("scale", 1.0))
    
    # Load model
    model = RandLANet(num_classes=3)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✓ Model loaded on device: {device}")
    
    # Load test data
    test_path = os.path.join(args.dataset_dir, args.test_file)
    if not os.path.exists(test_path):
        print(f"❌ Test file not found: {test_path}")
        print(f"   Available tests: test_randla_net.las, val_randla_net.las")
        return
    
    print(f"✓ Loading test data: {args.test_file}")
    points, labels = load_points_and_labels(args.dataset_dir, file_names=[args.test_file])
    print(f"  Total points: {points.shape[0]}")
    print(f"  Label distribution: C0={np.sum(labels==0)}, C1={np.sum(labels==1)}, C2={np.sum(labels==2)}")
    
    # Normalize
    points = (points - center) / max(scale, 1e-8)
    
    # Create dataset
    test_ds = RandomPointBlockDataset(
        points,
        labels,
        num_points=args.num_points,
        steps_per_epoch=max(100, points.shape[0] // args.num_points),
        augment=False,
    )
    
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, 
                                              shuffle=False, num_workers=0)
    
    # Evaluate
    print("\n" + "="*50)
    print("Testing...")
    print("="*50)
    
    total_tp = total_tn = total_fp = total_fn = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for i, (xyz, y) in enumerate(test_loader):
            xyz = xyz.to(device)
            y = y.to(device)
            
            logits = model(xyz)
            pred = torch.argmax(logits, dim=1)
            
            tp, tn, fp, fn = confusion_binary(pred, y)
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"  Batch {i + 1}/{len(test_loader)}")
    
    # Compute metrics
    metrics = binary_metrics(total_tp, total_tn, total_fp, total_fn)
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"IoU:       {metrics['iou']:.4f}")
    print("="*50 + "\n")
    
    # Compute per-class metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    print("Per-class Metrics:")
    for cls in range(3):
        class_name = ["Background", "Power Line", "Pylon"][cls]
        class_pred = (predictions == cls)
        class_true = (targets == cls)
        
        tp = np.sum(class_pred & class_true)
        fp = np.sum(class_pred & ~class_true)
        fn = np.sum(~class_pred & class_true)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        print(f"  {class_name:12} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Plot results
    plot_results(predictions, targets, metrics)
    

def plot_results(predictions, targets, metrics):
    """Plot confusion matrix and metrics"""
    from sklearn.metrics import confusion_matrix
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(targets, predictions, labels=[0, 1, 2])
    class_names = ["Background", "Power Line", "Pylon"]
    
    ax = axes[0, 0]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', 
                   color='white' if cm[i, j] > cm.max() / 2 else 'black',
                   fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    # Metrics Bar Chart
    ax = axes[0, 1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                    metrics['f1'], metrics['iou']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylim([0, 1.0])
    ax.set_title('Overall Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Per-class Precision/Recall/F1
    ax = axes[1, 0]
    per_class_metrics = {'Precision': [], 'Recall': [], 'F1': []}
    
    for cls in range(3):
        class_pred = (predictions == cls)
        class_true = (targets == cls)
        
        tp = np.sum(class_pred & class_true)
        fp = np.sum(class_pred & ~class_true)
        fn = np.sum(~class_pred & class_true)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        per_class_metrics['Precision'].append(precision)
        per_class_metrics['Recall'].append(recall)
        per_class_metrics['F1'].append(f1)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    ax.bar(x - width, per_class_metrics['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, per_class_metrics['Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, per_class_metrics['F1'], width, label='F1-Score', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Per-Class Metrics', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Class Distribution
    ax = axes[1, 1]
    unique, counts = np.unique(targets, return_counts=True)
    class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar([class_names[i] for i in unique], counts, color=class_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as: test_results.png")
    plt.show()


if __name__ == "__main__":
    main()
