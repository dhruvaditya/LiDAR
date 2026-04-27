import laspy
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from pathlib import Path

# ------------------------------
# PARAMETERS (TUNE IF NEEDED)
# ------------------------------
HEIGHT_THRESHOLD = 3.0
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 30

LINEARITY_THRESHOLD = 0.85
VERTICAL_THRESHOLD = 0.85
HORIZONTAL_THRESHOLD = 0.3

MIN_CLUSTER_SIZE = 80
MAX_RESIDUAL = 0.15   # RANSAC fitting error threshold

# ------------------------------
# LOAD DATA
# ------------------------------
BASE = Path(__file__).resolve().parents[2]
las_path = BASE / "Dataset" / "Classified_Electrical_Pole & Line.las"

las = laspy.read(str(las_path))
xyz = np.vstack((las.x, las.y, las.z)).T

# Normalize height
xyz[:, 2] -= np.min(xyz[:, 2])

labels = np.zeros(len(xyz), dtype=np.uint8)

# ------------------------------
# STEP 1: FILTER HIGH OBJECTS
# ------------------------------
high_idx = np.where(xyz[:, 2] > HEIGHT_THRESHOLD)[0]
xyz_high = xyz[high_idx]

# ------------------------------
# STEP 2: CLUSTERING
# ------------------------------
clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(xyz_high)
cluster_labels = clustering.labels_

# ------------------------------
# STEP 3: PROCESS EACH CLUSTER
# ------------------------------
for cluster_id in set(cluster_labels):

    if cluster_id == -1:
        continue

    cluster_indices = high_idx[cluster_labels == cluster_id]

    if len(cluster_indices) < MIN_CLUSTER_SIZE:
        continue

    pts = xyz[cluster_indices]

    # ------------------------------
    # PCA ANALYSIS
    # ------------------------------
    pca = PCA(n_components=3)
    pca.fit(pts)

    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_

    linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
    direction = eigenvectors[0]

    # ------------------------------
    # RANSAC LINE FITTING
    # ------------------------------
    try:
        X = pts[:, :2]  # use XY
        y = pts[:, 2]   # predict Z

        ransac = RANSACRegressor(residual_threshold=MAX_RESIDUAL)
        ransac.fit(X, y)

        inlier_mask = ransac.inlier_mask_
        residual = np.mean(np.abs(y[inlier_mask] - ransac.predict(X[inlier_mask])))

    except:
        continue

    # ------------------------------
    # CLASSIFICATION LOGIC
    # ------------------------------

    # POWER LINE (horizontal + linear + low residual)
    if (
        linearity > LINEARITY_THRESHOLD and
        abs(direction[2]) < HORIZONTAL_THRESHOLD and
        residual < MAX_RESIDUAL
    ):
        labels[cluster_indices] = 1

    # POLE (vertical + linear)
    elif (
        linearity > LINEARITY_THRESHOLD and
        abs(direction[2]) > VERTICAL_THRESHOLD
    ):
        labels[cluster_indices] = 2

# ------------------------------
# STEP 4: CLEAN SMALL NOISE
# ------------------------------
for class_id in [1, 2]:
    idx = np.where(labels == class_id)[0]

    if len(idx) == 0:
        continue

    sub_cluster = DBSCAN(eps=0.2, min_samples=20).fit(xyz[idx])
    sub_labels = sub_cluster.labels_

    for cid in set(sub_labels):
        if cid == -1:
            labels[idx[sub_labels == -1]] = 0
        else:
            pts_idx = idx[sub_labels == cid]
            if len(pts_idx) < 50:
                labels[pts_idx] = 0

# ------------------------------
# SAVE OUTPUT
# ------------------------------
output_path = BASE / "Dataset" / "train_labeled_randlanet.las"

# Ensure directory exists
output_path.parent.mkdir(parents=True, exist_ok=True)

# Create new LAS file
new_las = laspy.create(
    point_format=las.header.point_format,
    file_version=las.header.version
)

# Copy coordinates
new_las.x = las.x
new_las.y = las.y
new_las.z = las.z

# Copy intensity if exists
if 'intensity' in las.point_format.dimension_names:
    new_las.intensity = las.intensity

# Assign classification
labels = labels.astype(np.uint8)
new_las.classification = labels

# Save file
print("Saving to:", output_path)
new_las.write(str(output_path))

print("File saved successfully!")

# ------------------------------
# REPORT
# ------------------------------
print("Labeling complete")
print("Power line points:", np.sum(labels == 1))
print("Pole points:", np.sum(labels == 2))

#Ok tested and it is sufficient to train the model. The code is now ready for submission.