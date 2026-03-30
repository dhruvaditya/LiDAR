import laspy
import numpy as np
from pathlib import Path

# ------------------------------
# PARAMETERS
# ------------------------------
BLOCK_SIZE = 10       # meters
MIN_POINTS = 1000     # skip small blocks

# ------------------------------
# PATHS
# ------------------------------
BASE = Path(__file__).resolve().parent
input_las = BASE / "data_split" / "train_randla_net.las"
output_dir = BASE / "data_split" / "randlanet_blocks"

output_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------
# LOAD DATA
# ------------------------------
las = laspy.read(str(input_las))

xyz = np.vstack((las.x, las.y, las.z)).T
labels = np.array(las.classification).astype(np.int32)

print("Loaded:", xyz.shape)

# ------------------------------
# NORMALIZE
# ------------------------------
xyz = xyz - np.mean(xyz, axis=0)

# ------------------------------
# BLOCK SPLITTING
# ------------------------------
min_x, min_y = np.min(xyz[:, 0]), np.min(xyz[:, 1])
max_x, max_y = np.max(xyz[:, 0]), np.max(xyz[:, 1])

block_id = 0

for x in np.arange(min_x, max_x, BLOCK_SIZE):
    for y in np.arange(min_y, max_y, BLOCK_SIZE):

        mask = (
            (xyz[:, 0] >= x) & (xyz[:, 0] < x + BLOCK_SIZE) &
            (xyz[:, 1] >= y) & (xyz[:, 1] < y + BLOCK_SIZE)
        )

        block_points = xyz[mask]
        block_labels = labels[mask]

        if len(block_points) < MIN_POINTS:
            continue

        # Combine
        block_data = np.hstack((
            block_points,
            block_labels.reshape(-1, 1)
        ))

        # Save
        np.save(output_dir / f"block_{block_id}.npy", block_data)

        block_id += 1

print("Total blocks created:", block_id)