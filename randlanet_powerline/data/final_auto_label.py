import laspy
import numpy as np

from pathlib import Path
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# ============================================================
# Parameters
# ============================================================

# ----- Input / Output -----
BASE = Path(__file__).resolve().parents[2]
LAS_PATH = BASE / "Dataset" / "Classified_Electrical_Pole & Line.las"
OUT_PATH = BASE / "Dataset" / "train_labeled_plus.las"

# ----- Neighborhood for local PCA -----
K_NEIGHBORS = 30

# ----- Height filtering -----
# Since ground is already removed, this only suppresses very low residual artifacts
MIN_HEIGHT_FOR_WIRE_CANDIDATE = 2.0
MIN_HEIGHT_FOR_POLE_CANDIDATE = 1.0

# ----- Initial pointwise candidate thresholds -----
WIRE_LINEARITY_MIN = 0.82
WIRE_VERTICALITY_MAX = 0.25        # principal direction should be mostly horizontal
WIRE_LOCAL_ZSPAN_MAX = 0.80        # wire neighborhood should not be too vertically spread

POLE_LINEARITY_MIN = 0.72
POLE_VERTICALITY_MIN = 0.88        # principal direction should be strongly vertical
POLE_LOCAL_XY_SPREAD_MAX = 0.35    # pole neighborhood should be narrow in XY

# ----- DBSCAN for candidate grouping -----
WIRE_DBSCAN_EPS = 0.45
WIRE_DBSCAN_MIN_SAMPLES = 15

POLE_DBSCAN_EPS = 0.30
POLE_DBSCAN_MIN_SAMPLES = 12

# ----- Wire cluster validation -----
WIRE_MIN_CLUSTER_POINTS = 40
WIRE_MIN_SPAN = 6.0                # minimum longitudinal span
WIRE_MAX_PERP_THICKNESS = 0.35     # cluster should be thin transverse to main direction
WIRE_MAX_PROFILE_RESIDUAL = 0.12   # RANSAC quadratic fit residual in vertical profile
WIRE_MIN_RANSAC_INLIER_RATIO = 0.70
WIRE_MAX_DIRECTION_Z = 0.30        # cluster main direction should remain horizontal-ish

# ----- Pole cluster validation -----
POLE_MIN_CLUSTER_POINTS = 30
POLE_MIN_HEIGHT = 3.0
POLE_MAX_RADIUS = 0.40             # pole should be compact in XY
POLE_MIN_SLENDERNESS = 8.0         # height / radius
POLE_MAX_DIRECTION_XY = 0.25       # pole principal axis should be nearly vertical

# ----- Cluster growing / label recovery -----
WIRE_GROW_RADIUS = 0.18
POLE_GROW_RADIUS = 0.16

# ----- Label IDs -----
LABEL_BACKGROUND = 0
LABEL_WIRE = 1
LABEL_POLE = 2


# ============================================================
# Utility functions
# ============================================================

def safe_linearity(evals):
    """
    evals are expected in descending order: l1 >= l2 >= l3
    """
    l1 = max(evals[0], 1e-12)
    return (evals[0] - evals[1]) / l1


def compute_local_features(xyz, k):
    """
    Compute local PCA features for each point.

    Returns:
        linearity      : (N,)
        verticality    : (N,)  absolute Z component of dominant direction
        local_z_span   : (N,)
        local_xy_spread: (N,)
    """
    tree = KDTree(xyz)
    N = xyz.shape[0]

    linearity = np.zeros(N, dtype=np.float32)
    verticality = np.zeros(N, dtype=np.float32)
    local_z_span = np.zeros(N, dtype=np.float32)
    local_xy_spread = np.zeros(N, dtype=np.float32)

    for i in range(N):
        idx = tree.query([xyz[i]], k=min(k, N), return_distance=False)[0]
        pts = xyz[idx]

        centered = pts - pts.mean(axis=0, keepdims=True)

        # PCA by covariance eigendecomposition
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)  # ascending
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        main_dir = evecs[:, 0]

        linearity[i] = safe_linearity(evals)
        verticality[i] = abs(main_dir[2])
        local_z_span[i] = pts[:, 2].max() - pts[:, 2].min()

        xy_centered = pts[:, :2] - pts[:, :2].mean(axis=0, keepdims=True)
        local_xy_spread[i] = np.sqrt(np.mean(np.sum(xy_centered**2, axis=1)))

    return linearity, verticality, local_z_span, local_xy_spread


def cluster_indices(points, eps, min_samples):
    """
    DBSCAN clustering on a given subset of points.

    Returns:
        cluster_labels: array of DBSCAN cluster labels
    """
    if len(points) == 0:
        return np.array([], dtype=int)
    return DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_


def fit_main_axis(points):
    """
    PCA-based dominant axis of a cluster.
    Returns centroid, direction, PCA eigenvalues
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    direction = evecs[:, 0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return centroid, direction, evals


def quadratic_ransac_profile_fit(points, main_dir):
    """
    Fit z = f(s) using quadratic RANSAC,
    where s is projection along the main horizontal direction.

    Returns:
        success, stats_dict
    """
    d = main_dir.copy()

    # Use only horizontal component as curve axis
    d_xy = d[:2]
    d_xy_norm = np.linalg.norm(d_xy)
    if d_xy_norm < 1e-8:
        return False, {}

    e_s = np.array([d_xy[0], d_xy[1], 0.0]) / d_xy_norm

    origin = points.mean(axis=0)
    centered = points - origin

    s = centered @ e_s
    z = points[:, 2]

    # Require enough span
    span = float(np.max(s) - np.min(s))
    if span < WIRE_MIN_SPAN:
        return False, {"span": span}

    # RANSAC quadratic fit z = a*s^2 + b*s + c
    X = s.reshape(-1, 1)
    y = z

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lin", LinearRegression())
    ])

    ransac = RANSACRegressor(
        estimator=model,
        min_samples=max(10, int(0.35 * len(points))),
        residual_threshold=WIRE_MAX_PROFILE_RESIDUAL,
        random_state=42
    )

    try:
        ransac.fit(X, y)
    except Exception:
        return False, {"span": span}

    inlier_mask = ransac.inlier_mask_
    if inlier_mask is None:
        return False, {"span": span}

    inlier_ratio = float(np.mean(inlier_mask))
    y_pred = ransac.predict(X)
    residuals = np.abs(y - y_pred)
    med_res = float(np.median(residuals))

    return True, {
        "span": span,
        "inlier_ratio": inlier_ratio,
        "median_residual": med_res,
        "inlier_mask": inlier_mask
    }


def validate_wire_cluster(points):
    """
    Validate whether a cluster is truly a power line.

    Criteria:
    - enough points
    - dominant direction mostly horizontal
    - long span
    - thin transverse thickness
    - quadratic RANSAC fit in vertical profile has low residual
    """
    if len(points) < WIRE_MIN_CLUSTER_POINTS:
        return False

    centroid, direction, evals = fit_main_axis(points)

    if abs(direction[2]) > WIRE_MAX_DIRECTION_Z:
        return False

    # Thinness transverse to horizontal axis
    d_xy = direction[:2]
    d_xy_norm = np.linalg.norm(d_xy)
    if d_xy_norm < 1e-8:
        return False

    e_s = np.array([d_xy[0], d_xy[1], 0.0]) / d_xy_norm
    centered = points - centroid

    # Transverse distance in XY to main axis
    proj = (centered @ e_s).reshape(-1, 1) * e_s.reshape(1, 3)
    residual = centered - proj
    perp_xy = np.linalg.norm(residual[:, :2], axis=1)
    thickness = float(np.percentile(perp_xy, 90))

    if thickness > WIRE_MAX_PERP_THICKNESS:
        return False

    ok, stats = quadratic_ransac_profile_fit(points, direction)
    if not ok:
        return False

    if stats["inlier_ratio"] < WIRE_MIN_RANSAC_INLIER_RATIO:
        return False

    if stats["median_residual"] > WIRE_MAX_PROFILE_RESIDUAL:
        return False

    return True


def validate_pole_cluster(points):
    """
    Validate whether a cluster is truly a pole.

    Criteria:
    - enough points
    - dominant direction nearly vertical
    - sufficient height
    - compact radius in XY
    - slender geometry
    """
    if len(points) < POLE_MIN_CLUSTER_POINTS:
        return False

    centroid, direction, evals = fit_main_axis(points)

    # Pole principal axis should be near vertical
    direction_xy = np.linalg.norm(direction[:2])
    if direction_xy > POLE_MAX_DIRECTION_XY:
        return False

    z_span = float(points[:, 2].max() - points[:, 2].min())
    if z_span < POLE_MIN_HEIGHT:
        return False

    xy = points[:, :2]
    xy_center = np.median(xy, axis=0)
    r = np.linalg.norm(xy - xy_center, axis=1)

    radius90 = float(np.percentile(r, 90))
    if radius90 > POLE_MAX_RADIUS:
        return False

    slenderness = z_span / max(radius90, 1e-6)
    if slenderness < POLE_MIN_SLENDERNESS:
        return False

    return True


def grow_labels_from_cluster(all_xyz, seed_points, radius):
    """
    Recover nearby unlabeled points belonging to an already validated cluster.
    Uses radius search from each cluster point.
    """
    if len(seed_points) == 0:
        return np.array([], dtype=np.int64)

    tree = KDTree(all_xyz)
    neighbor_lists = tree.query_radius(seed_points, r=radius)

    if len(neighbor_lists) == 0:
        return np.array([], dtype=np.int64)

    idx = np.unique(np.concatenate(neighbor_lists))
    return idx.astype(np.int64)


# ============================================================
# Main
# ============================================================

def main():
    print("Reading LAS...")
    las = laspy.read(str(LAS_PATH))
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    # Normalize Z
    xyz[:, 2] -= np.min(xyz[:, 2])

    N = len(xyz)
    labels = np.zeros(N, dtype=np.uint8)

    print("Computing local PCA features...")
    linearity, verticality, local_z_span, local_xy_spread = compute_local_features(
        xyz, K_NEIGHBORS
    )

    # --------------------------------------------------------
    # Stage 1: Initial pointwise candidates
    # --------------------------------------------------------
    print("Generating initial candidates...")

    wire_candidate_mask = (
        (xyz[:, 2] >= MIN_HEIGHT_FOR_WIRE_CANDIDATE) &
        (linearity >= WIRE_LINEARITY_MIN) &
        (verticality <= WIRE_VERTICALITY_MAX) &
        (local_z_span <= WIRE_LOCAL_ZSPAN_MAX)
    )

    pole_candidate_mask = (
        (xyz[:, 2] >= MIN_HEIGHT_FOR_POLE_CANDIDATE) &
        (linearity >= POLE_LINEARITY_MIN) &
        (verticality >= POLE_VERTICALITY_MIN) &
        (local_xy_spread <= POLE_LOCAL_XY_SPREAD_MAX)
    )

    wire_candidate_idx = np.where(wire_candidate_mask)[0]
    pole_candidate_idx = np.where(pole_candidate_mask)[0]

    print(f"Initial wire candidates: {len(wire_candidate_idx)}")
    print(f"Initial pole candidates: {len(pole_candidate_idx)}")

    # --------------------------------------------------------
    # Stage 2: Cluster wire candidates and validate
    # --------------------------------------------------------
    print("Clustering wire candidates...")
    validated_wire_indices = []

    if len(wire_candidate_idx) > 0:
        wire_cluster_labels = cluster_indices(
            xyz[wire_candidate_idx],
            eps=WIRE_DBSCAN_EPS,
            min_samples=WIRE_DBSCAN_MIN_SAMPLES
        )

        for cid in np.unique(wire_cluster_labels):
            if cid == -1:
                continue

            cluster_global_idx = wire_candidate_idx[wire_cluster_labels == cid]
            cluster_points = xyz[cluster_global_idx]

            if validate_wire_cluster(cluster_points):
                grown_idx = grow_labels_from_cluster(xyz, cluster_points, WIRE_GROW_RADIUS)
                validated_wire_indices.append(grown_idx)

    if len(validated_wire_indices) > 0:
        validated_wire_indices = np.unique(np.concatenate(validated_wire_indices))
        labels[validated_wire_indices] = LABEL_WIRE
    else:
        validated_wire_indices = np.array([], dtype=np.int64)

    print(f"Validated wire points after growth: {len(validated_wire_indices)}")

    # --------------------------------------------------------
    # Stage 3: Cluster pole candidates and validate
    # --------------------------------------------------------
    print("Clustering pole candidates...")
    validated_pole_indices = []

    if len(pole_candidate_idx) > 0:
        pole_cluster_labels = cluster_indices(
            xyz[pole_candidate_idx],
            eps=POLE_DBSCAN_EPS,
            min_samples=POLE_DBSCAN_MIN_SAMPLES
        )

        for cid in np.unique(pole_cluster_labels):
            if cid == -1:
                continue

            cluster_global_idx = pole_candidate_idx[pole_cluster_labels == cid]
            cluster_points = xyz[cluster_global_idx]

            if validate_pole_cluster(cluster_points):
                grown_idx = grow_labels_from_cluster(xyz, cluster_points, POLE_GROW_RADIUS)
                validated_pole_indices.append(grown_idx)

    if len(validated_pole_indices) > 0:
        validated_pole_indices = np.unique(np.concatenate(validated_pole_indices))
        # poles overwrite wires near pole-wire intersection
        labels[validated_pole_indices] = LABEL_POLE
    else:
        validated_pole_indices = np.array([], dtype=np.int64)

    print(f"Validated pole points after growth: {len(validated_pole_indices)}")

    # --------------------------------------------------------
    # Optional post-cleaning:
    # Remove tiny leftover connected components in each class
    # --------------------------------------------------------
    print("Post-cleaning labels...")
    for class_id, eps, min_samples, min_keep in [
        (LABEL_WIRE, 0.25, 10, 30),
        (LABEL_POLE, 0.20, 8, 20),
    ]:
        idx = np.where(labels == class_id)[0]
        if len(idx) == 0:
            continue

        cl = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz[idx]).labels_
        for cid in np.unique(cl):
            if cid == -1:
                labels[idx[cl == -1]] = LABEL_BACKGROUND
                continue

            pts_idx = idx[cl == cid]
            if len(pts_idx) < min_keep:
                labels[pts_idx] = LABEL_BACKGROUND

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------
    print("Writing output LAS...")
    las.classification = labels
    las.write(str(OUT_PATH))

    print("Done.")
    print("Background points:", int(np.sum(labels == LABEL_BACKGROUND)))
    print("Wire points      :", int(np.sum(labels == LABEL_WIRE)))
    print("Pole points      :", int(np.sum(labels == LABEL_POLE)))


if __name__ == "__main__":
    main()