import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.colors import hsv_to_rgb

def sample_vertices(k, seed=42, equal_norm=True, radius=1.0) -> tuple[np.ndarray, ConvexHull]:
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, size=k)
    r = np.full(k, radius) if equal_norm else rng.uniform(0.4, 1.0, size=k)
    V = r[:, None] * np.c_[np.cos(theta), np.sin(theta)]
    hull = ConvexHull(V)
    return V, hull

def compute_cell_map(grid: np.ndarray, basis: np.ndarray, inside: np.ndarray) -> np.ndarray:
    cell_map = np.argmax(grid @ basis.T, axis=1)
    cell_map[~inside] = -1
    return cell_map

def plot_cells_simple(k=5, i_target=2, eta=1e-3, seed=42):
    V, hull = sample_vertices(k, seed=seed, equal_norm=True)
    K = V[hull.vertices]

    pad = 0.1
    xmin, ymin = K.min(axis=0) - pad
    xmax, ymax = K.max(axis=0) + pad
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 1000), np.linspace(ymin, ymax, 1000))
    grid = np.c_[xx.ravel(), yy.ravel()]
    inside = Path(K).contains_points(grid)

    basis = K
    cell_map = compute_cell_map(grid, basis, inside)

    plt.figure(figsize=(8, 8))

    # Step 1: Draw all cells in white with black outlines
    for i in range(len(K)):
        mask = (cell_map == i)
        if not np.any(mask):
            continue
        pts = grid[mask]
        try:
            ch = ConvexHull(pts)
            hull_coords = pts[ch.vertices]
            color = [0.95, 0.95, 0.95] if i == i_target else 'white'
            plt.gca().add_patch(MplPolygon(hull_coords, facecolor=color, edgecolor='black', linewidth=0.5))
        except:
            pass

    # Step 2: Draw strict dominance cone in red inside i_target
    pts = grid[cell_map == i_target]
    v_i = K[i_target]
    others = [K[j] for j in range(len(K)) if j != i_target]
    diffs = np.stack([v_i - vj for vj in others])
    inner = pts @ diffs.T
    mask_cone = np.all(inner > eta, axis=1)
    pts_cone = pts[mask_cone]
    if len(pts_cone) > 0:
        try:
            ch_cone = ConvexHull(pts_cone)
            hull_cone = pts_cone[ch_cone.vertices]
            plt.gca().add_patch(MplPolygon(hull_cone, facecolor=hsv_to_rgb([1/3, 1, 1]), edgecolor='black', linewidth=0.5, alpha=0.6))
        except:
            pass

    # Step 3: Plot polygon and vertices
    plt.plot(*K[np.r_[np.arange(len(K)), 0]].T, 'k', lw=1)
    plt.scatter(K[:, 0], K[:, 1], facecolors='none', edgecolors='black', s=80, linewidths=1.5)
    plt.scatter(0, 0, color='black', marker='x', s=60, linewidths=2)

    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()

    import os
    current_dir = os.getcwd()
    filename = os.path.join(current_dir, f"dominance_cells_k{k}.pdf")
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Saved plot to {filename}")

# Example usage
if __name__ == "__main__":
    plot_cells_simple(k=7, i_target=2, eta=0.5*1e-1, seed=42)