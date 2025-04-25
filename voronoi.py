import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os

def construct_finite_voronoi_regions(vor, radius=10):
    """Construct finite Voronoi regions by closing unbounded ones."""
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    all_ridges = {}

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if -1 not in region:
            new_regions.append(region)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in region if v != -1]

        for p2, v1, v2 in ridges:
            if v2 == -1:
                v1, v2 = v2, v1
            if v1 != -1:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

# # Generate k random points
np.random.seed(42)
# k = 18
# V_all = np.random.rand(k, 2) - 0.5  # Center around origin
# print(V_all)

from scipy.linalg import fractional_matrix_power

# Settings
k = 5
c = 1.0
B = np.diag([1, 1])  # example positive definite matrix
Binv_sqrt = fractional_matrix_power(B, -0.5)  # B^{-1/2}

# Random angles
theta = np.random.uniform(0, 2 * np.pi, k)
circle_points = np.c_[np.cos(theta), np.sin(theta)]  # shape (k, 2)

# Map to ellipse level set: x = sqrt(c) * B^{-1/2} * u
V_all = np.sqrt(c) * (circle_points @ Binv_sqrt.T)

# Keep only convex hull vertices
hull = ConvexHull(V_all)
V = V_all[hull.vertices]
hull_poly = ShapelyPolygon(V)

# Compute Voronoi on hull vertices
vor = Voronoi(V)
regions, vertices = construct_finite_voronoi_regions(vor)

# Set up plot
fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(V)))
patches = []
region_colors = []

# Intersect Voronoi cells with convex hull
for i, region in enumerate(regions):
    polygon = vertices[region]
    poly = ShapelyPolygon(polygon)
    if not poly.is_valid:
        continue
    clipped = hull_poly.intersection(poly)
    if not clipped.is_empty:
        for geom in clipped.geoms if isinstance(clipped, MultiPolygon) else [clipped]:
            coords = np.array(geom.exterior.coords)
            patches.append(Polygon(coords, closed=True))
            region_colors.append(colors[i])

# Plot Voronoi cells
if patches:
    collection = PatchCollection(patches, facecolor=region_colors, alpha=0.5, edgecolor='k', linewidth=0.5)
    ax.add_collection(collection)

# Plot convex hull
ax.plot(*V[[*range(len(V)), 0]].T, 'k', lw=1)

# Plot convex hull vertices
ax.scatter(V[:, 0], V[:, 1], facecolors='none', edgecolors='black', s=80, linewidths=1.5)
for i, v in enumerate(V):
    ax.text(v[0] + 0.01, v[1] + 0.01, f'$v_{{{i+1}}}$', fontsize=13)

# Plot origin
plt.scatter(0, 0, color='black', marker='x', s=80, linewidths=2)
plt.text(0.02, 0.02, "$0$", color='black', fontsize=13)

ax.set_title("Voronoi Cells of Convex Hull Vertices (Clipped to Hull)")
ax.set_aspect('equal')
ax.axis('off')
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "voronoi_{}.png".format(k))
plt.savefig(filename, dpi=500)
plt.show()