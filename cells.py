import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from matplotlib import rc
import os

# LaTeX settings
rc("text", usetex=True)
rc("font", size=13)


np.random.seed(42)

# Generate random points
k = 36
V = np.random.rand(k, 2) - 0.5  # Center around origin

# # Generate points on a circle with small radial noise
# k = 9
# theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
# radii = 1 + 0.2 * (np.random.rand(k) - 0.5)  # small radial noise
# V = np.c_[radii * np.cos(theta), radii * np.sin(theta)]

from scipy.linalg import fractional_matrix_power

# Settings
# k = 5
# c = 1.0
B = np.diag([1, 1])  # example positive definite matrix
# Binv_sqrt = fractional_matrix_power(B, -0.5)  # B^{-1/2}

# # Random angles
# theta = np.random.uniform(0, 2 * np.pi, k)
# circle_points = np.c_[np.cos(theta), np.sin(theta)]  # shape (k, 2)

# # Map to ellipse level set: x = sqrt(c) * B^{-1/2} * u
# V = np.sqrt(c) * (circle_points @ Binv_sqrt.T)



#B = np.diag([1, 1])  # Example: scales x-component by 1, y-component by 2


# Compute convex hull and extract hull vertices
hull = ConvexHull(V)
K = V[hull.vertices]  # Points on the convex hull
vertex_labels = hull.vertices  # Indices into V

# Grid over bounding box
xmin, ymin = K.min(axis=0) - 0.1
xmax, ymax = K.max(axis=0) + 0.1
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 1000), np.linspace(ymin, ymax, 1000))
grid = np.c_[xx.ravel(), yy.ravel()]

# Keep only grid points inside convex hull
path = Path(K)
inside = path.contains_points(grid)

# Compute dominance cells
cell_map = -np.ones(grid.shape[0], dtype=int)
for i, z in enumerate(grid):
    if inside[i]:
        scores = K @ (B @ z)  # inner products <z, v_j>
        cell_map[i] = np.argmax(scores)

# Plot
from matplotlib.colors import hsv_to_rgb
plt.figure(figsize=(8, 8))
hues = np.linspace(0, 1, len(K), endpoint=False)  # evenly spaced hues
saturation = 1.0
value = 1.0
colors = hsv_to_rgb(np.stack([hues, np.full(len(K), saturation), np.full(len(K), value)], axis=1))
#colors = plt.cm.tab10(np.linspace(0, 1, len(K)))


for i in range(len(K)):
    mask = (cell_map == i)
    if np.any(mask):
        points = grid[mask]
        plt.scatter(points[:, 0], points[:, 1], color=colors[i], s=1)

        # Label cell with original index from V
        centroid = points.mean(axis=0)
        original_index = vertex_labels[i]
        plt.text(centroid[0], centroid[1], fr"$\mathcal{{C}}(v_{{{original_index+1}}})$", fontsize=13,
                 ha='center', va='center')

# Draw convex hull
plt.plot(*K[[*range(len(K)), 0]].T, 'k', lw=1)

# Plot convex hull vertices as hollow circles
plt.scatter(K[:, 0], K[:, 1], facecolors='none', edgecolors='black', s=80, linewidths=1.5)
offset_scale = 0.05  # tweak as needed
centroid_K = K.mean(axis=0)

for i, v in enumerate(K):
    idx = vertex_labels[i]
    direction = v - centroid_K
    normal = direction / np.linalg.norm(direction)
    offset = offset_scale * normal
    label_pos = v + offset

    plt.text(label_pos[0], label_pos[1], f"$v_{{{idx+1}}}$", fontsize=13,
             ha='center', va='center')

# Plot the origin
plt.scatter(0, 0, color='black', marker='x', s=80, linewidths=2)
plt.text(0.02, 0.02, "$0$", color='black', fontsize=13)

#plt.title("Cells in $\mathcal{K}$ where $\\langle x, v_j \\rangle \\geq \\langle x, y \\rangle$ for all $y \\in \mathcal{K}$")
plt.axis('equal')
plt.axis('off')
script_dir = os.path.dirname(os.path.abspath(__file__))
#filename = os.path.join(script_dir, "cells{}.png".format(k))
filename = os.path.join(script_dir, "cells{}.pdf".format(k))
plt.savefig(filename, bbox_inches='tight', format='pdf')
plt.show()