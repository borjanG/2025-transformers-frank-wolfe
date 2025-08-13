import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from matplotlib import rc
from scipy.linalg import fractional_matrix_power
import os

# --- LaTeX and font settings for publication-quality figures ---
rc("text", usetex=True)
rc("font", size=13)

def sample_vertices(
    k: int,
    *,
    seed: int | None = None,
    equal_norm: bool = False,
    radius: float = 1.0,
    radial_range: tuple[float, float] = (0.4, 1.0),
    enforce_bijection: bool = False,
    enforce_self_inclusion: bool = False,
    max_tries: int = 500,
) -> tuple[np.ndarray, ConvexHull]:
    """
    Generate k random points in R² subject to optional constraints.

    Returns
    -------
    V     : (k,2) ndarray of sampled points
    hull  : scipy.spatial.ConvexHull(V)
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        theta = rng.uniform(0, 2 * np.pi, size=k)
        r = np.full(k, radius) if equal_norm else rng.uniform(*radial_range, size=k)
        V = r[:, None] * np.c_[np.cos(theta), np.sin(theta)]
        hull = ConvexHull(V)

        if enforce_bijection and len(hull.vertices) != k:
            continue

        if enforce_self_inclusion:
            ok = True
            for i, vi in enumerate(V):
                if any(vj @ vi > vi @ vi + 1e-14 for j, vj in enumerate(V) if j != i):
                    ok = False
                    break
            if not ok:
                continue

        return V, hull

# --- Parameters ---
k = 5
seed = 42
c = 1.0
B = np.diag([1, 1])                          # Quadratic form matrix
Binv_sqrt = fractional_matrix_power(B, -0.5)  # Compute B^{-1/2}

# --- Sample points on distorted circle ---
circle_points, hull = sample_vertices(k, seed=seed, equal_norm=True)
V = np.sqrt(c) * (circle_points @ Binv_sqrt.T)  # Map to ellipse

# --- Convex hull of mapped points ---
K = V[hull.vertices]

# --- Grid over bounding box of convex hull ---
margin = 0.1
xmin, ymin = K.min(axis=0) - margin
xmax, ymax = K.max(axis=0) + margin
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
grid = np.c_[xx.ravel(), yy.ravel()]

# --- Determine which points are inside the convex hull ---
path = Path(K)
inside = path.contains_points(grid)

# --- Evaluate quadratic form f(x) = 0.5 xᵀ B x ---
f_vals = 0.5 * np.einsum('ij,jk,ik->i', grid, B, grid)
f_masked = np.full_like(xx, np.nan, dtype=np.float64)
f_masked.ravel()[inside] = f_vals[inside]

# --- Plotting ---
plt.figure(figsize=(8, 8))

# Contour plot
cp = plt.contourf(xx, yy, f_masked, levels=30, cmap='plasma')
cbar = plt.colorbar(cp, orientation='horizontal', fraction=0.05, pad=0.08)
cbar.set_label(r"$\frac{1}{2} \langle Bx, x \rangle$")

# Convex hull boundary
plt.plot(*K[[*range(len(K)), 0]].T, 'k-', lw=1)

# Sample points
plt.scatter(V[:, 0], V[:, 1], color='black', marker='x')

#plt.title(r"$x \mapsto \frac{1}{2} \langle Bx, x \rangle$ over $\mathcal{K}$")
plt.axis('equal')
plt.axis('off')
plt.tight_layout()

# --- Save as PDF ---
script_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(script_dir, f"quadratic_{k}.pdf")
plt.savefig(filename, dpi=500, bbox_inches='tight')
plt.show()
