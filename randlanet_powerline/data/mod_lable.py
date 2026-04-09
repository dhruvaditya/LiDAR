import laspy
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Parameters - Tuned for precision
# ------------------------------
HEIGHT_THRESHOLD = 2.0              # Lower threshold to catch more candidates
K_NEIGHBORS = 30                     # Slightly more neighbors for stable PCA
LINEARITY_THRESHOLD = 0.75           # Slightly relaxed for robustness
PLANARITY_THRESHOLD = 0.3            # For scattering detection
HORIZONTAL_Z_THRESHOLD = 0.25        # Power lines: nearly horizontal
VERTICAL_Z_THRESHOLD = 0.85          # Poles: nearly vertical

# Geometric validation parameters
MIN_POLE_HEIGHT = 4.0                # Minimum pole height in meters
MAX_POLE_DIAMETER = 0.8              # Maximum expected pole diameter
MIN_WIRE_LENGTH = 5.0                # Minimum wire span length
RANSAC_RESIDUAL_THRESHOLD = 0.15       # For line fitting (meters)
MIN_RANSAC_INLIERS_RATIO = 0.7         # At least 70% must fit the model

# Clustering parameters
DBSCAN_EPS_WIRE = 0.3                # Wires are more spread out
DBSCAN_EPS_POLE = 0.15               # Poles are compact
DBSCAN_MIN_SAMPLES = 15
MIN_CLUSTER_SIZE_WIRE = 30
MIN_CLUSTER_SIZE_POLE = 40

BASE = Path(__file__).resolve().parents[2]
las_path = BASE / "Dataset" / "Classified_Electrical_Pole & Line.las"

print("Loading data...")
las = laspy.read(str(las_path))
xyz = np.vstack((las.x, las.y, las.z)).T

# Normalize height
z_min = np.min(xyz[:,2])
xyz[:,2] -= z_min
print(f"Point count: {len(xyz)}, Height range: {xyz[:,2].min():.2f} to {xyz[:,2].max():.2f}")

tree = KDTree(xyz)
labels = np.zeros(len(xyz), dtype=np.uint8)
candidate_scores = np.zeros(len(xyz), dtype=np.float32)  # Confidence scores

# ------------------------------
# Stage 1: PCA Feature Extraction & Initial Classification
# ------------------------------
print("Stage 1: PCA feature extraction...")
for i in range(len(xyz)):
    if xyz[i,2] < HEIGHT_THRESHOLD:
        continue

    # Adaptive k-neighbors based on local density (optional refinement)
    neighbors_idx = tree.query([xyz[i]], k=K_NEIGHBORS, return_distance=False)[0]
    neighbors = xyz[neighbors_idx]
    
    # Skip if neighbors are too far (edge of dataset)
    if np.max(np.linalg.norm(neighbors - xyz[i], axis=1)) > 2.0:
        continue

    pca = PCA(n_components=3)
    pca.fit(neighbors - neighbors.mean(axis=0))  # Centered PCA
    
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    
    # Normalize eigenvalues
    total_variance = np.sum(eigenvalues)
    if total_variance < 1e-6:
        continue
        
    e1, e2, e3 = eigenvalues / total_variance
    
    # Geometric features
    linearity = (e1 - e2) / (e1 + 1e-10)           # 1-D characteristic
    planarity = (e2 - e3) / (e1 + 1e-10)           # 2-D characteristic
    scattering = e3 / (e1 + 1e-10)                 # 3-D characteristic (noise)
    
    main_direction = eigenvectors[0]
    verticalness = abs(main_direction[2])
    horizontalness = np.sqrt(main_direction[0]**2 + main_direction[1]**2)

    # Store features for debugging (optional)
    # Scoring system for confidence
    confidence = 0.0

    # -------------------------
    # Power Line Detection Logic
    # -------------------------
    # Must be: linear, horizontal, not planar (thin), elevated
    is_linear = linearity > LINEARITY_THRESHOLD
    is_horizontal = verticalness < HORIZONTAL_Z_THRESHOLD
    is_thin = planarity < 0.5  # Wires are thin, not planar like ground
    
    if is_linear and is_horizontal and is_thin and xyz[i,2] > 3.0:
        # Additional check: neighbors should form a line pattern
        # Check if points spread mainly in XY plane along main direction
        proj_xy = neighbors[:, :2] - xyz[i, :2]
        direction_xy = main_direction[:2] / (np.linalg.norm(main_direction[:2]) + 1e-10)
        
        # Project onto main direction in XY
        projections = proj_xy @ direction_xy
        spread_along = np.std(projections)
        spread_across = np.std(np.linalg.norm(proj_xy - np.outer(projections, direction_xy), axis=1))
        
        # Wires have high spread along direction, low spread across
        if spread_along > 0.5 and spread_across < 0.3:
            labels[i] = 1
            candidate_scores[i] = linearity * (1 - verticalness) * 100

    # -------------------------
    # Pole Detection Logic
    # -------------------------
    # Must be: linear, vertical, cylindrical (specific planarity pattern)
    is_vertical = verticalness > VERTICAL_Z_THRESHOLD
    
    if is_linear and is_vertical and xyz[i,2] > 1.0:
        # Check cylindrical characteristics: spread in XY should be small and uniform
        xy_spread = np.std(neighbors[:, :2], axis=0)
        if np.max(xy_spread) < MAX_POLE_DIAMETER / 2 and np.min(xy_spread) > 0.02:
            # Check vertical extent of neighbors
            z_spread = np.max(neighbors[:, 2]) - np.min(neighbors[:, 2])
            if z_spread > 0.5:  # Must have vertical extent
                labels[i] = 2
                candidate_scores[i] = linearity * verticalness * 100

print(f"Initial detection - Wires: {np.sum(labels==1)}, Poles: {np.sum(labels==2)}")

# ------------------------------
# Stage 2: Clustering & Geometric Validation
# ------------------------------
print("Stage 2: Clustering and geometric validation...")

def validate_wire_cluster(cluster_xyz):
    """
    Validate wire cluster using RANSAC line fitting.
    Wires should fit to 3D lines with low residual.
    """
    if len(cluster_xyz) < MIN_CLUSTER_SIZE_WIRE:
        return False, 0.0
    
    # Try to fit a 3D line using RANSAC
    # Parameterize line as point + direction
    try:
        # Center the cluster
        centroid = np.mean(cluster_xyz, axis=0)
        centered = cluster_xyz - centroid
        
        # Use PCA to get initial direction
        pca = PCA(n_components=3)
        pca.fit(centered)
        direction = pca.components_[0]
        
        # Project all points onto the line
        projections = centered @ direction
        closest_points_on_line = np.outer(projections, direction)
        residuals = np.linalg.norm(centered - closest_points_on_line, axis=1)
        
        # Check residuals
        inliers = residuals < RANSAC_RESIDUAL_THRESHOLD
        inlier_ratio = np.sum(inliers) / len(residuals)
        
        # Additional check: line should be roughly horizontal
        vertical_component = abs(direction[2])
        
        # Wires should be long enough
        line_length = np.max(projections) - np.min(projections)
        
        valid = (inlier_ratio > MIN_RANSAC_INLIERS_RATIO and 
                vertical_component < HORIZONTAL_Z_THRESHOLD and
                line_length > MIN_WIRE_LENGTH)
        
        return valid, inlier_ratio
        
    except Exception as e:
        return False, 0.0

def validate_pole_cluster(cluster_xyz):
    """
    Validate pole cluster using cylindrical fitting.
    Poles should be vertical cylinders with consistent diameter.
    """
    if len(cluster_xyz) < MIN_CLUSTER_SIZE_POLE:
        return False, 0.0
    
    try:
        # Check vertical extent
        z_range = np.max(cluster_xyz[:, 2]) - np.min(cluster_xyz[:, 2])
        if z_range < MIN_POLE_HEIGHT:
            return False, 0.0
        
        # Check if vertical (mean direction should be Z)
        pca = PCA(n_components=3)
        pca.fit(cluster_xyz - np.mean(cluster_xyz, axis=0))
        
        # Main component should be vertical
        verticalness = abs(pca.components_[0, 2])
        if verticalness < VERTICAL_Z_THRESHOLD:
            return False, 0.0
        
        # Check cylindrical shape: project to XY plane, should form circle
        xy_points = cluster_xyz[:, :2]
        xy_center = np.mean(xy_points, axis=0)
        radii = np.linalg.norm(xy_points - xy_center, axis=1)
        
        radius_mean = np.mean(radii)
        radius_std = np.std(radii)
        
        # Consistent radius (cylinder) and reasonable diameter
        cv = radius_std / (radius_mean + 1e-10)  # Coefficient of variation
        diameter = 2 * radius_mean
        
        valid = (cv < 0.4 and  # Consistent radius
                diameter < MAX_POLE_DIAMETER and 
                diameter > 0.05 and
                verticalness > 0.9)
        
        return valid, 1.0 - cv
        
    except Exception as e:
        return False, 0.0

# Process wire clusters
wire_indices = np.where(labels == 1)[0]
if len(wire_indices) > 0:
    print(f"Processing {len(wire_indices)} wire candidates...")
    clustering = DBSCAN(eps=DBSCAN_EPS_WIRE, min_samples=DBSCAN_MIN_SAMPLES).fit(xyz[wire_indices])
    cluster_labels = clustering.labels_
    
    unique_labels = set(cluster_labels)
    for cluster_id in unique_labels:
        if cluster_id == -1:
            # Noise
            mask = cluster_labels == -1
            labels[wire_indices[mask]] = 0
        else:
            mask = cluster_labels == cluster_id
            cluster_points = wire_indices[mask]
            
            valid, score = validate_wire_cluster(xyz[cluster_points])
            if not valid:
                labels[cluster_points] = 0
            else:
                # Keep as wire, optionally store confidence
                pass

# Process pole clusters
pole_indices = np.where(labels == 2)[0]
if len(pole_indices) > 0:
    print(f"Processing {len(pole_indices)} pole candidates...")
    clustering = DBSCAN(eps=DBSCAN_EPS_POLE, min_samples=DBSCAN_MIN_SAMPLES).fit(xyz[pole_indices])
    cluster_labels = clustering.labels_
    
    unique_labels = set(cluster_labels)
    for cluster_id in unique_labels:
        if cluster_id == -1:
            mask = cluster_labels == -1
            labels[pole_indices[mask]] = 0
        else:
            mask = cluster_labels == cluster_id
            cluster_points = pole_indices[mask]
            
            valid, score = validate_pole_cluster(xyz[cluster_points])
            if not valid:
                labels[cluster_points] = 0
            else:
                # Keep as pole
                pass

print(f"After geometric validation - Wires: {np.sum(labels==1)}, Poles: {np.sum(labels==2)}")

# ------------------------------
# Stage 3: Contextual Refinement
# ------------------------------
print("Stage 3: Contextual refinement...")

# Refinement 1: Wires should be near poles (connectivity check)
# Find pole top positions
pole_indices = np.where(labels == 2)[0]
if len(pole_indices) > 0 and np.sum(labels == 1) > 0:
    pole_tops = []
    clustering = DBSCAN(eps=DBSCAN_EPS_POLE, min_samples=DBSCAN_MIN_SAMPLES).fit(xyz[pole_indices])
    
    for cid in set(clustering.labels_):
        if cid == -1:
            continue
        mask = clustering.labels_ == cid
        cluster_pts = xyz[pole_indices[mask]]
        pole_tops.append(np.max(cluster_pts[:, 2]))  # Top Z of pole
    
    # Wires too far above all pole tops might be false positives
    wire_indices = np.where(labels == 1)[0]
    if len(pole_tops) > 0:
        max_pole_top = max(pole_tops)
        # Wires should be within reasonable range of pole tops
        # But power lines can sag, so allow some margin
        too_high_mask = xyz[wire_indices, 2] > (max_pole_top + 2.0)
        if np.sum(too_high_mask) > 0:
            labels[wire_indices[too_high_mask]] = 0

# Refinement 2: Remove isolated small clusters that survived
for class_id in [1, 2]:
    class_indices = np.where(labels == class_id)[0]
    if len(class_indices) == 0:
        continue
    
    eps = DBSCAN_EPS_WIRE if class_id == 1 else DBSCAN_EPS_POLE
    clustering = DBSCAN(eps=eps, min_samples=10).fit(xyz[class_indices])
    
    for cid in set(clustering.labels_):
        if cid == -1:
            continue
        mask = clustering.labels_ == cid
        if np.sum(mask) < (MIN_CLUSTER_SIZE_WIRE if class_id == 1 else MIN_CLUSTER_SIZE_POLE):
            labels[class_indices[mask]] = 0

# Refinement 3: Pole-wire spatial relationship
# Wires should generally run between poles (optional advanced check)
if len(pole_indices) > 1 and np.sum(labels == 1) > 0:
    # Build KD-tree of poles for wire proximity check
    pole_centroids = []
    clustering = DBSCAN(eps=DBSCAN_EPS_POLE, min_samples=DBSCAN_MIN_SAMPLES).fit(xyz[pole_indices])
    
    for cid in set(clustering.labels_):
        if cid == -1:
            continue
        mask = clustering.labels_ == cid
        pole_centroids.append(np.mean(xyz[pole_indices[mask]], axis=0))
    
    if len(pole_centroids) > 1:
        pole_centroids = np.array(pole_centroids)
        pole_tree = KDTree(pole_centroids[:, :2])  # 2D tree in XY
        
        wire_indices = np.where(labels == 1)[0]
        wire_xy = xyz[wire_indices, :2]
        
        # Check if wires are within reasonable distance of any pole
        distances, _ = pole_tree.query(wire_xy, k=1)
        # Power lines typically within 15m of poles (adjust as needed)
        far_from_poles = distances > 20.0
        
        if np.sum(far_from_poles) > 0:
            # Don't remove immediately, just mark for review or lower confidence
            # Only remove if also small cluster
            pass

print(f"After contextual refinement - Wires: {np.sum(labels==1)}, Poles: {np.sum(labels==2)}")

# ------------------------------
# Stage 4: Final Cleanup
# ------------------------------
print("Stage 4: Final cleanup...")

# Ensure no overlap between classes (shouldn't happen, but safety check)
double_labeled = (labels == 1) & (labels == 2)  # Impossible but logic check
if np.sum(double_labeled) > 0:
    labels[double_labeled] = 0

# Remove any remaining isolated points
for class_id in [1, 2]:
    mask = labels == class_id
    if np.sum(mask) < 5:  # Too few points to be meaningful
        labels[mask] = 0

# ------------------------------
# Save Results
# ------------------------------
print("Saving results...")
new_las = laspy.create(point_format=las.header.point_format,
                       file_version=las.header.version)

new_las.points = las.points
new_las.classification = labels

# Optional: Add intensity or user_data field for confidence scores
# new_las.user_data = candidate_scores.astype(np.uint8)

output_path = BASE / "Dataset" / "train_labeled_kimclass_refined.las"
new_las.write(str(output_path))

print("\n" + "="*50)
print("LABELING COMPLETE")
print("="*50)
print(f"Total points processed: {len(xyz):,}")
print(f"Power line points (1): {np.sum(labels==1):,} ({100*np.sum(labels==1)/len(xyz):.2f}%)")
print(f"Pole points (2): {np.sum(labels==2):,} ({100*np.sum(labels==2)/len(xyz):.2f}%)")
print(f"Unlabeled (0): {np.sum(labels==0):,} ({100*np.sum(labels==0)/len(xyz):.2f}%)")
print(f"Output saved to: {output_path}")
print("="*50)

# Generate detailed statistics for verification
print("\nDetailed Statistics:")
wire_indices = np.where(labels == 1)[0]
pole_indices = np.where(labels == 2)[0]

if len(wire_indices) > 0:
    print(f"\nWire clusters height range: {xyz[wire_indices, 2].min():.2f} to {xyz[wire_indices, 2].max():.2f}m")
if len(pole_indices) > 0:
    print(f"Pole clusters height range: {xyz[pole_indices, 2].min():.2f} to {xyz[pole_indices, 2].max():.2f}m")