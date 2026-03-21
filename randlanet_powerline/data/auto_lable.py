import laspy
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from pathlib import Path

# ------------------------------
# Parameters (tune if needed)
# ------------------------------
HEIGHT_THRESHOLD = 4.0     # meters above ground
K_NEIGHBORS = 20
LINEARITY_THRESHOLD = 0.85

# ------------------------------
# Load LAS
# ------------------------------
BASE = Path(__file__).resolve().parents[2]  # project root: Power_Line
las_path = BASE / "Dataset" / "Classified_Electrical_Pole & Line.las"
las = laspy.read(str(las_path))

xyz = np.vstack((las.x, las.y, las.z)).T

# Normalize height
z_min = np.min(xyz[:, 2])
xyz[:, 2] -= z_min

# ------------------------------
# Step 1: Height Filtering
# ------------------------------
high_points_mask = xyz[:, 2] > HEIGHT_THRESHOLD

print("Points above height threshold:", np.sum(high_points_mask))

# ------------------------------
# Step 2: Local Linearity via PCA
# ------------------------------
tree = KDTree(xyz)

labels = np.zeros(len(xyz), dtype=np.uint8)

candidate_indices = np.where(high_points_mask)[0]

for idx in candidate_indices:

    neighbors_idx = tree.query([xyz[idx]], k=K_NEIGHBORS, return_distance=False)[0]
    neighbors = xyz[neighbors_idx]

    pca = PCA(n_components=3)
    pca.fit(neighbors)

    eigenvalues = pca.explained_variance_

    # Linearity measure
    linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]

    if linearity > LINEARITY_THRESHOLD:
        labels[idx] = 1  # Mark as wire

# ------------------------------
# Save New LAS with Labels
# ------------------------------
new_las = laspy.create(point_format=las.header.point_format,
                       file_version=las.header.version)

new_las.points = las.points
new_las.classification = labels

new_las.write(str(BASE / "Dataset" / "train_labeled.las"))

print("Auto-labeling complete.")
print("Wire points detected:", np.sum(labels == 1))