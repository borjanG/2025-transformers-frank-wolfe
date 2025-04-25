import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from matplotlib import rc

# LaTeX settings
rc("text", usetex=True)
rc("font", size=13)

# Generate random points
np.random.seed(42)
#k = 9
#V = np.random.rand(k, 2) - 0.5  # Center around origin

# k = 9
# theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
# radii = 1 + 0.2 * (np.random.rand(k) - 0.5)  # small radial noise
# V = np.c_[radii * np.cos(theta), radii * np.sin(theta)]

# # Matrix B
# B = np.diag([1, 2])  # Example quadratic form

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
V = np.sqrt(c) * (circle_points @ Binv_sqrt.T)

# Convex hull of V
hull = ConvexHull(V)
K = V[hull.vertices]
vertex_labels = hull.vertices

# Grid over bounding box
xmin, ymin = K.min(axis=0) - 0.1
xmax, ymax = K.max(axis=0) + 0.1
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
grid = np.c_[xx.ravel(), yy.ravel()]

# Mask points outside convex hull
path = Path(K)
inside = path.contains_points(grid)

# Evaluate quadratic form on grid
f_vals = np.einsum('ij,jk,ik->i', grid, B, grid) * 0.5  # 0.5 * x^T B x
f_masked = np.full_like(xx, np.nan, dtype=np.float64)
f_masked.ravel()[inside] = f_vals[inside]

# Plot
plt.figure(figsize=(8, 8))
cp = plt.contourf(xx, yy, f_masked, levels=30, cmap='plasma')
cbar = plt.colorbar(cp, orientation='horizontal', fraction=0.05, pad=0.08)
cbar.set_label(r"$\frac{1}{2} \langle Bx, x \rangle$")


# Draw convex hull boundary
plt.plot(*K[[*range(len(K)), 0]].T, 'k-', lw=1)

# Optional: plot original points
plt.scatter(V[:, 0], V[:, 1], color='black', marker='x')

plt.title(r"$x \mapsto \frac{1}{2} \langle Bx, x \rangle$ over $\mathcal{K}$")
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, "quadratic_{}.png".format(k))
plt.savefig(filename, dpi=500)
plt.show()
