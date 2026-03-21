from pathlib import Path

import laspy
BASE = Path(__file__).resolve().parents[2]  # project root: Power_Line
las_path = BASE / "Dataset" / "Classified_Electrical_Pole & Line.las"
las = laspy.read(str(las_path))
print(las_path)