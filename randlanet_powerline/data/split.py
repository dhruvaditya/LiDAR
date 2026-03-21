import laspy
import numpy as np
from pathlib import Path

# las = laspy.read("Dataset\Classified_Electrical_Pole & Line.las")
BASE = Path(__file__).resolve().parents[2]  # project root: Power_Line
las_path = BASE / "Dataset" / "Classified_Electrical_Pole & Line.las"
las = laspy.read(str(las_path))

xyz = np.vstack((las.x, las.y, las.z)).T

x_min = np.min(xyz[:,0])
x_max = np.max(xyz[:,0])

x_range = x_max - x_min

train_limit = x_min + 0.75 * x_range
val_limit = x_min + 0.90 * x_range

train_idx = xyz[:,0] <= train_limit
val_idx = (xyz[:,0] > train_limit) & (xyz[:,0] <= val_limit)
test_idx = xyz[:,0] > val_limit

# Save subsets
def save_subset(mask, filename):
    new_las = laspy.create(point_format=las.header.point_format,
                           file_version=las.header.version)
    new_las.points = las.points[mask]
    new_las.write(filename)

save_subset(train_idx, "train.las")
save_subset(val_idx, "val.las")
save_subset(test_idx, "test.las")

print("Spatial split complete.")

# OK Tested for spatial split of the dataset into train, val, and test subsets