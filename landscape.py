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

k = 9
theta = np.linspace(0, 2 * np.pi, k, endpoint=False)
radii = 1 + 0.2 * (np.random.rand(k) - 0.5)  # small radial noise
V = np.c_[radii * np.cos(theta), radii * np.sin(theta)]

# Matrix B for the bilinear form <Bx, y>
B = np.diag([1, 2])  # Change as needed

# Convex hull of V
hull = ConvexHull(V)
K = V[hull.vertices]            # Convex hull vertices
vertex_labels = hull.vertices   # Indices into V

# Create a grid over the bounding box of K
xmin, ymin = K.min(axis=0) - 0.1
xmax, ymax = K.max(axis=0) + 0.1
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 400), np.linspace(ymin, ymax, 400))
grid = np.c_[xx.ravel(), yy.ravel()]

# Mask out points outside the convex hull
path = Path(K)
inside = path.contains_points(grid)

# Plotting
plt.figure(figsize=(12, 12))

for i, x in enumerate(V):
    u = B @ x  # Vector to define <Bx, y>
    fx = grid @ u  # Evaluate <Bx, y> over the grid
    fx_masked = np.full_like(xx, np.nan, dtype=np.float64)
    fx_masked.ravel()[inside] = fx[inside]

    # Subplot for each x
    plt.subplot(3, 3, i + 1)
    cp = plt.contourf(xx, yy, fx_masked, levels=20, cmap='coolwarm')
    plt.colorbar(cp, shrink=0.8)
    plt.title(f"$f_{{x_{i+1}}}(y) = \\langle Bx_{{{i+1}}}, y \\rangle$")
    
    # Draw convex hull
    plt.plot(*K[[*range(len(K)), 0]].T, 'k-', lw=1)
    
    # Mark point x
    plt.scatter(*x, color='black', marker='x', zorder=3)
    
    plt.axis('equal')
    plt.axis('off')

plt.tight_layout()
plt.savefig("Desktop/contours_over_K.png", dpi=400)
plt.show()
