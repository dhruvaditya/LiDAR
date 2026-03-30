# Testing & Visualization Guide

## Quick Start

### Step 1: Train the Model
```bash
cd c:\Users\User\Desktop\Aditya
& lidar\Scripts\Activate.ps1

# Train on pre-split data
python -m randlanet_powerline.train `
    --dataset_dir randlanet_powerline/data/data_split `
    --epochs 50 `
    --batch_size 4 `
    --save_dir checkpoints
```

**Expected output:**
```
Epoch 001 | train_loss=X.XXXX | val_loss=X.XXXX | val_iou=X.XXXX | val_f1=X.XXXX
Epoch 002 | train_loss=X.XXXX | val_loss=X.XXXX | val_iou=X.XXXX | val_f1=X.XXXX
...
Training complete. Best validation IoU: X.XXXX
```

Checkpoints saved to: `checkpoints/best.pt` (best model) and `checkpoints/last.pt` (latest)

---

### Step 2: Test & Visualize Metrics

#### Install scikit-learn (required for plotting):
```bash
pip install scikit-learn
```

#### Run test with visualization:
```bash
python test_and_plot.py --checkpoint checkpoints/best.pt
```

**This will:**
- Load the trained model
- Test on `test_randla_net.las`
- Print detailed metrics (Accuracy, Precision, Recall, F1, IoU)
- Generate `test_results.png` with 4 plots:
  1. **Confusion Matrix** - Shows per-class predictions
  2. **Overall Metrics** - Bar chart of all global metrics
  3. **Per-Class Metrics** - Precision, Recall, F1 for each class
  4. **Class Distribution** - Sample counts in test set

---

### Step 3: Alternative - Evaluate on Validation Set
```bash
python -m randlanet_powerline.evaluate `
    --checkpoint checkpoints/best.pt `
    --dataset_dir randlanet_powerline/data/data_split
```

---

## Full Workflow Example

```bash
# 1. Activate environment
& c:\Users\User\Desktop\Aditya\lidar\Scripts\Activate.ps1
cd c:\Users\User\Desktop\Aditya

# 2. Train model
python -m randlanet_powerline.train `
    --dataset_dir randlanet_powerline/data/data_split `
    --epochs 50 `
    --batch_size 4

# 3. Test and visualize
python test_and_plot.py --checkpoint checkpoints/best.pt
```

---

## Optional: Test on Different Sets

### Test on Validation Set:
```bash
python test_and_plot.py `
    --checkpoint checkpoints/best.pt `
    --test_file val_randla_net.las
```

### Test on Training Set (check for overfitting):
```bash
python test_and_plot.py `
    --checkpoint checkpoints/best.pt `
    --test_file train_randla_net.las
```

---

## Understanding the Metrics

- **Accuracy**: Overall correctness (TP+TN) / Total
- **Precision**: How many predicted power lines are actually power lines TP / (TP+FP)
- **Recall**: How many actual power lines were detected TP / (TP+FN)
- **F1-Score**: Harmonic mean of precision and recall
- **IoU (Intersection over Union)**: TP / (TP+FP+FN) - Best for imbalanced data

## Troubleshooting

**If checkpoint not found:**
- Verify training completed successfully
- Check `checkpoints/` folder exists
- Ensure you ran training first

**If test data not found:**
- Verify file exists in: `randlanet_powerline/data/data_split/`
- Files should be: `train_randla_net.las`, `val_randla_net.las`, `test_randla_net.las`

**If evaluation is slow:**
- Reduce batch size: `--batch_size 2`
- Reduce num_points: `--num_points 2048`

