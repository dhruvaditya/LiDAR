import laspy
import numpy as np
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]
las_path = BASE / "Dataset" / "train_labeled_randlanet.las"
def inspect_las_metadata(las_path):
    """Comprehensive LAS file metadata inspection"""
    las = laspy.read(las_path)
    
    print("=" * 60)
    print("LAS FILE METADATA REPORT")
    print("=" * 60)
    
    # Basic file info
    print(f"\n📁 File: {las_path}")
    print(f"📝 Point Format: {las.header.point_format}")
    print(f"🔢 Point Format ID: {las.header.point_format.id}")
    print(f"📊 File Version: {las.header.version}")
    print(f"🔢 Number of Points: {len(las.points):,}")
    
    # Available dimensions (critical for RandLA-Net)
    print(f"\n📋 Available Dimensions:")
    standard_dims = list(las.point_format.standard_dimension_names)
    extra_dims = list(las.point_format.extra_dimension_names)
    
    print(f"   Standard ({len(standard_dims)}): {standard_dims}")
    if extra_dims:
        print(f"   Extra ({len(extra_dims)}): {extra_dims}")
    
    # Check for classification labels (essential for training)
    print(f"\n🏷️  CLASSIFICATION ANALYSIS:")
    if hasattr(las, 'classification'):
        unique_labels, counts = np.unique(las.classification, return_counts=True)
        print(f"   Unique Labels: {unique_labels}")
        print(f"   Label Distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(las.classification)) * 100
            print(f"      Class {label}: {count:,} points ({percentage:.2f}%)")
        
        # Check for unlabeled points (usually class 0 or 1)
        unlabeled_count = np.sum(las.classification == 0)
        if unlabeled_count > 0:
            print(f"   ⚠️  WARNING: {unlabeled_count:,} unlabeled points (class 0)")
    else:
        print("   ❌ No classification field found!")
    
    # Coordinate bounds
    print(f"\n🌐 COORDINATE BOUNDS:")
    print(f"   X: {las.header.x_min:.3f} to {las.header.x_max:.3f}")
    print(f"   Y: {las.header.y_min:.3f} to {las.header.y_max:.3f}")
    print(f"   Z: {las.header.z_min:.3f} to {las.header.z_max:.3f}")
    
    # Check for additional features useful for RandLA-Net
    print(f"\n🔍 POTENTIAL FEATURES FOR RANDLA-NET:")
    features = {}
    
    if hasattr(las, 'intensity'):
        features['intensity'] = (las.intensity.min(), las.intensity.max())
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        features['rgb'] = True
    if hasattr(las, 'return_number'):
        features['return_number'] = np.unique(las.return_number)
    if hasattr(las, 'number_of_returns'):
        features['number_of_returns'] = np.unique(las.number_of_returns)
    if hasattr(las, 'scan_angle_rank'):
        features['scan_angle'] = (las.scan_angle_rank.min(), las.scan_angle_rank.max())
    if hasattr(las, 'gps_time'):
        features['gps_time'] = True
    if hasattr(las, 'user_data'):
        features['user_data'] = True
        
    for feat, value in features.items():
        print(f"   ✓ {feat}: {value}")
    
    # Data quality checks
    print(f"\n⚡ DATA QUALITY CHECKS:")
    xyz = np.vstack((las.x, las.y, las.z)).T
    
    # Check for duplicate points
    unique_points = np.unique(xyz, axis=0)
    if len(unique_points) < len(xyz):
        print(f"   ⚠️  Duplicate points: {len(xyz) - len(unique_points):,}")
    
    # Check for NaN/Inf values
    nan_count = np.sum(np.isnan(xyz))
    inf_count = np.sum(np.isinf(xyz))
    if nan_count > 0 or inf_count > 0:
        print(f"   ❌ NaN: {nan_count}, Inf: {inf_count}")
    else:
        print(f"   ✓ No NaN or Inf values")
    
    print("=" * 60)
    
    return {
        'las': las,
        'xyz': xyz,
        'labels': las.classification if hasattr(las, 'classification') else None,
        'features': features,
        'num_classes': len(unique_labels) if hasattr(las, 'classification') else 0
    }

# Usage
metadata = inspect_las_metadata(las_path)