import laspy
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from pathlib import Path

# ------------------------------
# PARAMETERS (TUNE BASED ON DATA)
# ------------------------------
HEIGHT_THRESHOLD = 2.5

DBSCAN_EPS = 0.4
DBSCAN_MIN_SAMPLES = 40

LINEARITY_THRESHOLD = 0.85
VERTICAL_THRESHOLD = 0.85
HORIZONTAL_THRESHOLD = 0.3

MIN_CLUSTER_SIZE = 100

# ------------------------------
# PATHS
# ------------------------------
BASE = Path(__file__).resolve().parent
input_las = BASE / "Dataset" / "Classified_Electrical_Pole & Line.las"
output_las = BASE / "Dataset" / "auto_labeled.las"

# ------------------------------
# LOAD DATA
# ------------------------------
las = laspy.read(str(input_las))

xyz = np.vstack((las.x, las.y, las.z)).T
xyz[:, 2] -= np.min(xyz[:, 2])

labels = np.zeros(len(xyz), dtype=np.uint8)

print("Total points:", len(xyz))

# ------------------------------
# STEP 1: HEIGHT FILTER
# ------------------------------
high_idx = np.where(xyz[:, 2] > HEIGHT_THRESHOLD)[0]
xyz_high = xyz[high_idx]

print("High points:", len(high_idx))

# ------------------------------
# STEP 2: DBSCAN CLUSTERING
# ------------------------------
clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(xyz_high)
cluster_labels = clustering.labels_

print("Clusters found:", len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0))

# ------------------------------
# STEP 3: PROCESS EACH CLUSTER
# ------------------------------
for cid in set(cluster_labels):

    if cid == -1:
        continue

    cluster_indices = high_idx[cluster_labels == cid]

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

    z_component = abs(direction[2])

    # ------------------------------
    # SHAPE FEATURES
    # ------------------------------
    z_range = np.max(pts[:, 2]) - np.min(pts[:, 2])
    xy_range = np.linalg.norm(
        np.max(pts[:, :2], axis=0) - np.min(pts[:, :2], axis=0)
    )

    # ------------------------------
    # CLASSIFICATION LOGIC
    # ------------------------------

    # 🔥 POWER LINE
    if (
        linearity > LINEARITY_THRESHOLD and
        z_component < HORIZONTAL_THRESHOLD and
        xy_range > 5 and               # long horizontal structure
        z_range < 2                    # small vertical variation
    ):
        labels[cluster_indices] = 1

    # 🔥 POLE
    elif (
        linearity > LINEARITY_THRESHOLD and
        z_component > VERTICAL_THRESHOLD and
        z_range > 3 and                # tall vertical structure
        xy_range < 2                   # narrow in XY
    ):
        labels[cluster_indices] = 2

# ------------------------------
# STEP 4: REMOVE NOISE
# ------------------------------
for class_id in [1, 2]:

    idx = np.where(labels == class_id)[0]

    if len(idx) == 0:
        continue

    sub_cluster = DBSCAN(eps=0.2, min_samples=20).fit(xyz[idx])
    sub_labels = sub_cluster.labels_

    for scid in set(sub_labels):
        if scid == -1:
            labels[idx[sub_labels == -1]] = 0
        else:
            pts_idx = idx[sub_labels == scid]
            if len(pts_idx) < 50:
                labels[pts_idx] = 0

# ------------------------------
# SAVE OUTPUT
# ------------------------------
output_las.parent.mkdir(parents=True, exist_ok=True)

new_las = laspy.create(
    point_format=las.header.point_format,
    file_version=las.header.version
)

# Copy coordinates
new_las.x = las.x
new_las.y = las.y
new_las.z = las.z

# Assign labels
new_las.classification = labels.astype(np.uint8)

# Save
new_las.write(str(output_las))

# ------------------------------
# REPORT
# ------------------------------
print("\n===== RESULT =====")
print("Power line points:", np.sum(labels == 1))
print("Pole points:", np.sum(labels == 2))
print("Unlabeled points:", np.sum(labels == 0))
print("Saved to:", output_las)

### Output of this code:
# Total points: 698272
# High points: 692926
# Clusters found: 543

# ===== RESULT =====
# Power line points: 21998
# Pole points: 26698
# Unlabeled points: 649576
###