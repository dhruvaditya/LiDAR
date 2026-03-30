import numpy as np
import torch


def confusion_binary(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.detach().cpu().numpy().astype(np.int64).reshape(-1)
    target = target.detach().cpu().numpy().astype(np.int64).reshape(-1)

    # For multi-class, compute per-class metrics
    tp = int(((pred == 1) & (target == 1)).sum())
    tn = int(((pred != 1) & (target != 1)).sum())  # True negatives (not class 1)
    fp = int(((pred == 1) & (target != 1)).sum())
    fn = int(((pred != 1) & (target == 1)).sum())
    return tp, tn, fp, fn


def binary_metrics(tp: int, tn: int, fp: int, fn: int):
    eps = 1e-8
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }
