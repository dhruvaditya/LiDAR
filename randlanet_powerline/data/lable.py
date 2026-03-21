import laspy
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # project root: Power_Line
las_path = BASE / "Dataset" / "Classified_Electrical_Pole & Line.las"
las = laspy.read(str(las_path))

# print("Point format:", las.header.point_format)
# print("Available dimensions:", list(las.point_format.dimension_names))
import numpy as np
print("Unique classification values:", np.unique(las.classification))