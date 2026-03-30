import laspy
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from pathlib import Path

# ------------------------------
# Optimized Parameters
# ------------------------------
K_NEIGHBORS = 30
LINEARITY_THRESHOLD = 0.75  # Lowered slightly to capture sag in wires
DBSCAN_EPS = 0.8            # Distance between points in a cluster
DBSCAN_MIN_SAMPLES = 10     
MIN_LINE_LENGTH = 5.0       # Minimum meters a power line cluster should span
MAX_RANSAC_RESIDUAL = 0.15  # Strictness of line fitting

def get_features(neighbors):
    """Calculate dimensionality features for a point neighborhood."""
    pca = PCA(n_components=3)
    pca.fit(neighbors)
    eig_vals = pca.explained_variance_
    
    # Avoid division by zero
    sum_vals = np.sum(eig_vals)
    if sum_vals == 0: return 0, 0, np.array([0,0,0])
    
    # Linearity (L), Planarity (P)
    l = (eig_vals[0] - eig_vals[1]) / eig_vals[0]
    p = (eig_vals[1] - eig_vals[2]) / eig_vals[0]
    
    return l, p, pca.components_[0]

def fit_ransac_line(points):
    """Uses RANSAC to check if a cluster is actually a linear structure."""
    if len(points) < 10: return False
    
    # Use X,Y to predict Z (or PCA for a more general 3D line)
    # For power lines, we mostly care about horizontal linearity
    X = points[:, :2] # X, Y
    y = points[:, 2]  # Z
    
    try:
        ransac = RANSACRegressor(residual_threshold=MAX_RANSAC_RESIDUAL)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        inlier_ratio = np.sum(inlier_mask) / len(points)
        return inlier_ratio > 0.7
    except:
        return False

# ------------------------------
# Execution
# ------------------------------
BASE = Path(__file__).resolve().parents[0] # Adjusted for local execution
las_path = BASE / "Dataset" / "Classified_Electrical_Pole & Line.las"

print("Loading LAS file...")
las = laspy.read(str(las_path))
xyz = np.vstack((las.x, las.y, las.z)).T
labels = np.zeros(len(xyz), dtype=np.uint8)

tree = KDTree(xyz)

print("Extracting geometric features...")
# Step 1: Initial Point-wise Geometry Classification
linearity_mask = np.zeros(len(xyz), dtype=bool)
pole_candidate_mask = np.zeros(len(xyz), dtype=bool)

for i in range(0, len(xyz), 2): # Step 2 for speed, or 1 for precision
    neighbors_idx = tree.query([xyz[i]], k=K_NEIGHBORS, return_distance=False)[0]
    l, p, main_dir = get_features(xyz[neighbors_idx])
    
    # Check if linear
    if l > LINEARITY_THRESHOLD:
        # Check orientation
        if abs(main_dir[2]) < 0.3: # Horizontal-ish
            linearity_mask[i] = True
        elif abs(main_dir[2]) > 0.7: # Vertical-ish
            pole_candidate_mask[i] = True

# Step 2: Cluster Refinement for Power Lines (Class 1)
print("Refining Power Lines...")
line_indices = np.where(linearity_mask)[0]
if len(line_indices) > 0:
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(xyz[line_indices])
    for cluster_id in set(db.labels_):
        if cluster_id == -1: continue
        
        idx_in_cluster = line_indices[db.labels_ == cluster_id]
        cluster_pts = xyz[idx_in_cluster]
        
        # Geometry validation
        extent = np.max(cluster_pts, axis=0) - np.min(cluster_pts, axis=0)
        horizontal_span = np.linalg.norm(extent[:2])
        
        # Power lines must have significant horizontal span
        if horizontal_span > MIN_LINE_LENGTH:
            if fit_ransac_line(cluster_pts):
                labels[idx_in_cluster] = 1

# Step 3: Cluster Refinement for Poles (Class 2)
print("Refining Poles...")
pole_indices = np.where(pole_candidate_mask)[0]
if len(pole_indices) > 0:
    db_pole = DBSCAN(eps=0.5, min_samples=15).fit(xyz[pole_indices])
    for cluster_id in set(db_pole.labels_):
        if cluster_id == -1: continue
        
        idx_in_cluster = pole_indices[db_pole.labels_ == cluster_id]
        cluster_pts = xyz[idx_in_cluster]
        
        extent = np.max(cluster_pts, axis=0) - np.min(cluster_pts, axis=0)
        
        # Poles must have more vertical extent than horizontal
        if extent[2] > 2.0 and np.max(extent[:2]) < 3.0:
            labels[idx_in_cluster] = 2

# ------------------------------
# Output
# ------------------------------
new_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
new_las.points = las.points
new_las.classification = labels
new_las.write(str(BASE / "Dataset" / "refined_labeled_output.las"))

print("-" * 30)
print(f"Labeling complete.")
print(f"Wire points (1): {np.sum(labels==1)}")
print(f"Pole points (2): {np.sum(labels==2)}")
print(f"Unclassified (0): {np.sum(labels==0)}")