import laspy
import numpy as np

# Load LAS file
las = laspy.read("Dataset/Merged clouds.las")

print("========== BASIC INFO ==========")
print(f"Points count: {len(las.points)}")
print(f"LAS version: {las.header.version}")
print(f"Point format: {las.header.point_format}")

print("\n========== DIMENSIONS ==========")
print(las.point_format.dimension_names)

print("\n========== STANDARD FIELDS ==========")
# Check classification (standard LAS field)
if 'classification' in las.point_format.dimension_names:
    classifications = las.classification
    unique_classes = np.unique(classifications)
    print("Unique classification values:", unique_classes)
else:
    print("No standard classification field found.")

print("\n========== EXTRA DIMENSIONS (SCALAR FIELDS) ==========")
# Check extra scalar fields
extra_dims = las.point_format.extra_dimension_names
print("Extra dimensions:", extra_dims)

for dim in extra_dims:
    data = las[dim]
    unique_vals = np.unique(data)
    print(f"{dim} → unique values: {unique_vals[:20]}")  # show first 20