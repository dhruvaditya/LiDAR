# Training Guide: 3-Class Classification (Power Line & Pylons)

## Overview
Updated the model to support 3 classes:
- **Class 0**: Background (other points)
- **Class 1**: Power Line
- **Class 2**: Pylon (electrical pole, tower, etc.)

## Changes Made

### 1. Data Labeling (`randlanet_powerline/data/las_dataset.py`)
- Updated to use `get_file_class()` function that assigns classes based on filename patterns
- **Power Line patterns**: "line", "power"
- **Pylon patterns**: "pylon", "pole", "electrical_pole", "tower"
- **Background**: Any file not matching above patterns

#### File Naming Convention Required:
Organize your LAS files with clear naming patterns:
- Power line files: Include `line` or `power` in filename (e.g., `powerline_area1.las`, `line_sector2.las`)
- Pylon files: Include `pylon`, `pole`, `electrical_pole`, or `tower` (e.g., `pylon_distribution.las`, `pole_cluster.las`)
- Background: Any other files (e.g., `background.las`, `unlabeled.las`)

### 2. Model Architecture (`randlanet_powerline/train.py`)
- Changed `RandLANet(num_classes=2)` → `RandLANet(num_classes=3)`
- Updated class weight computation to handle 3 classes

### 3. Updated Files
- **train.py**: Updated num_classes and class weights
- **infer.py**: Updated model initialization to use 3 classes
- **evaluate.py**: Updated model initialization to use 3 classes
- **metrics.py**: Updated confusion matrix for multi-class settings
- **las_dataset.py**: Updated labeling functions and class weight computation

## Dataset Preparation

### Step 1: Organize Your LAS Files
Your data should be in the `randlanet_powerline/data/data_split/` folder with pre-split and pre-labeled files:

```
randlanet_powerline/data/data_split/
├── train_randla_net.las    # Training data with labels in classification field
├── val_randla_net.las      # Validation data with labels in classification field  
└── test_randla_net.las     # Test data with labels in classification field
```

**Important**: The labels (0, 1, 2) must already be stored in the `classification` field of each LAS file.

### Step 2: Verify Data Labels
The model now reads labels directly from the LAS file's classification field, so ensure your data is properly labeled:
- **0**: Background points
- **1**: Power line points  
- **2**: Pylon points

### Step 3: Data Conversion (if needed)
If you need to convert or prepare your data:
```bash
cd randlanet_powerline/data
python data_conversion.py
```

## Training

### Basic Training Command:
```bash
cd c:\Users\User\Desktop\Aditya
python -m randlanet_powerline.train --dataset_dir randlanet_powerline/data/data_split --epochs 40 --batch_size 4
```

### Optional Parameters:
```bash
python -m randlanet_powerline.train \
    --dataset_dir randlanet_powerline/data/data_split \
    --epochs 50 \
    --batch_size 8 \
    --num_points 4096 \
    --lr 1e-3 \
    --save_dir checkpoints
```

**Note**: Training will use only `train_randla_net.las`. If `val_randla_net.las` exists, it will be used for validation. Otherwise, a portion of training data will be used for validation.

## Evaluation

After training, evaluate the model:
```bash
python -m randlanet_powerline.evaluate --checkpoint checkpoints/best.pt --dataset_dir randlanet_powerline/data/data_split
```

## Inference

Run inference on new LAS files:
```bash
python -m randlanet_powerline.infer \
    --checkpoint checkpoints/model_epoch_40.pt \
    --input_las input_cloud.las \
    --output_las predictions.las
```

## Output Format

The output predictions will have values 0, 1, or 2:
- **0**: Background
- **1**: Power Line
- **2**: Pylon

These are typically stored in the classification field of the output LAS file.

## Class Imbalance Handling

The updated model automatically computes class weights based on data distribution. If one class is significantly underrepresented:
- Use `--max_points_per_file` to balance sampling
- Ensure roughly equal representation of all 3 classes
- Monitor training/validation metrics to detect imbalance issues

## Tips for Best Results

1. **Naming Convention**: Be consistent with file naming patterns (case-insensitive)
   - Split files into meaningful groups
   - Avoid ambiguous naming (e.g., don't use "line" in pylon filenames)

2. **Data Balance**: Aim for roughly equal distribution of all three classes
   - If unbalanced, the weighted loss will adjust automatically
   - Monitor class distribution in training logs

3. **Validation Split**: Default is 80/20 train/val split
   - Adjust with `--val_ratio` if needed

4. **Checkpoint Management**: Models are saved per epoch
   - Choose the best checkpoint based on validation metrics
   - Use best F1-score or validation loss as selection criterion

## Troubleshooting

### Model Not Learning:
- Check that files are properly named and classified
- Verify data distribution (print class counts)
- Start with higher learning rate or more epochs

### Memory Issues:
- Reduce `--batch_size` (e.g., from 8 to 4)
- Reduce `--num_points` (e.g., from 4096 to 2048)

### All Points Classified as One Class:
- Verify file naming patterns match expectations
- Check class distribution across dataset
- Ensure sufficient training data for each class

