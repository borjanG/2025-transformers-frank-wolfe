import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path

def construct_finite_voronoi_regions(vor, radius=10):
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

def plot_voronoi_clipped(V, hull_poly, filename):
    vor = Voronoi(V)
    regions, vertices = construct_finite_voronoi_regions(vor)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(V)))
    patches, region_colors = [], []

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

    if patches:
        collection = PatchCollection(patches, facecolor=region_colors, alpha=0.5, edgecolor='k', linewidth=0.5)
        ax.add_collection(collection)

    ax.plot(*V[[*range(len(V)), 0]].T, 'k--', lw=1)
    ax.scatter(V[:, 0], V[:, 1], facecolors='none', edgecolors='black', s=80, linewidths=1.5)
    for i, v in enumerate(V):
        ax.text(v[0] + 0.01, v[1] + 0.01, f'$v_{{{i+1}}}$', fontsize=12)

    ax.scatter(0, 0, facecolors='none', edgecolors='blue', s=80, linewidths=2)
    ax.text(0.01, 0.01, '$0$', color='blue', fontsize=12)

    ax.set_title("Voronoi Cells Clipped to Convex Hull")
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_inner_product_cells(V, hull_poly, filename):
    xmin, ymin = V.min(axis=0) - 0.1
    xmax, ymax = V.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 1000), np.linspace(ymin, ymax, 1000))
    grid = np.c_[xx.ravel(), yy.ravel()]

    path = Path(hull_poly.exterior.coords)
    inside = path.contains_points(grid)

    colors = plt.cm.Set3(np.linspace(0, 1, len(V)))
    cell_map = -np.ones(grid.shape[0], dtype=int)

    for i, z in enumerate(grid):
        if inside[i]:
            scores = V @ z
            cell_map[i] = np.argmax(scores)

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(len(V)):
        mask = (cell_map == i)
        if np.any(mask):
            ax.scatter(grid[mask, 0], grid[mask, 1], color=colors[i], alpha=0.3, s=1)

    ax.plot(*V[[*range(len(V)), 0]].T, 'k--', lw=1)
    ax.scatter(V[:, 0], V[:, 1], facecolors='none', edgecolors='black', s=80, linewidths=1.5)
    for i, v in enumerate(V):
        ax.text(v[0]+0.01, v[1]+0.01, f'$v_{{{i+1}}}$', fontsize=12)

    ax.scatter(0, 0, facecolors='none', edgecolors='blue', s=80, linewidths=2)
    ax.text(0.01, 0.01, '$0$', color='blue', fontsize=12)

    ax.set_title("Inner Product Dominance Cells in Convex Hull")
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# === Main Execution ===
np.random.seed(42)
k = 6
V_all = np.random.rand(k, 2) - 0.5
hull = ConvexHull(V_all)
V_hull = V_all[hull.vertices]
hull_poly = ShapelyPolygon(V_hull)

# Save both figures
plot_voronoi_clipped(V_hull, hull_poly, "voronoi_cells.pdf")
plot_inner_product_cells(V_hull, hull_poly, "inner_product_cells.pdf")
