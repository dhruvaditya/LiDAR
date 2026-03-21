import laspy
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from pathlib import Path

# ------------------------------
# Parameters
# ------------------------------
HEIGHT_THRESHOLD = 3.5
K_NEIGHBORS = 25
LINEARITY_THRESHOLD = 0.8
HORIZONTAL_Z_THRESHOLD = 0.3
VERTICAL_Z_THRESHOLD = 0.8

BASE = Path(__file__).resolve().parents[2]
las_path = BASE / "Dataset" / "Classified_Electrical_Pole & Line.las"

las = laspy.read(str(las_path))
xyz = np.vstack((las.x, las.y, las.z)).T

# Normalize height
xyz[:,2] -= np.min(xyz[:,2])

tree = KDTree(xyz)
labels = np.zeros(len(xyz), dtype=np.uint8)

# ------------------------------
# PCA Feature Extraction
# ------------------------------
for i in range(len(xyz)):

    if xyz[i,2] < HEIGHT_THRESHOLD:
        continue

    neighbors_idx = tree.query([xyz[i]], k=K_NEIGHBORS, return_distance=False)[0]
    neighbors = xyz[neighbors_idx]

    pca = PCA(n_components=3)
    pca.fit(neighbors)

    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_

    linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]

    main_direction = eigenvectors[0]

    # ------------------------------
    # Power Line Detection
    # ------------------------------
    if linearity > LINEARITY_THRESHOLD and abs(main_direction[2]) < HORIZONTAL_Z_THRESHOLD:
        labels[i] = 1  # Power line

    # ------------------------------
    # Pole Detection
    # ------------------------------
    elif linearity > LINEARITY_THRESHOLD and abs(main_direction[2]) > VERTICAL_Z_THRESHOLD:
        labels[i] = 2  # Pole

# ------------------------------
# Cluster refinement
# ------------------------------

# Remove tiny clusters (noise)
for class_id in [1,2]:

    class_indices = np.where(labels == class_id)[0]

    if len(class_indices) == 0:
        continue

    clustering = DBSCAN(eps=0.2, min_samples=20).fit(xyz[class_indices])

    cluster_labels = clustering.labels_

    for cluster_id in set(cluster_labels):

        if cluster_id == -1:
            labels[class_indices[cluster_labels == -1]] = 0
        else:
            cluster_points = class_indices[cluster_labels == cluster_id]
            if len(cluster_points) < 50:
                labels[cluster_points] = 0

# ------------------------------
# Save
# ------------------------------
new_las = laspy.create(point_format=las.header.point_format,
                       file_version=las.header.version)

new_las.points = las.points
new_las.classification = labels

new_las.write(str(BASE / "Dataset" / "train_labeled_3class.las"))

print("Labeling complete")
print("Wire points:", np.sum(labels==1))
print("Pole points:", np.sum(labels==2))