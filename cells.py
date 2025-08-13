#!/usr/bin/env python3
"""
Random polytope ↔ dominance‑cell visualiser.

Four optional constraints:
  • equal_norm – vertices lie on common circle
  • enforce_bijection – |hull| = k
  • normalize_inner_product – use unit directions in arg‑max
  • enforce_self_inclusion – resample until each vᵢ lies in its own cell

Region labels 𝒞ᵢ(v) are drawn only when k ≤ 5.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Polygon as MplPolygon
from matplotlib import rc
import os

# ------------------------------------------------------------
# Global plotting defaults
# ------------------------------------------------------------
rc("text", usetex=True)
rc("font", size=13)
rc("text.latex", preamble=r"\usepackage{mathrsfs}")

# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------
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
    V     : (k,2) ndarray
    hull  : scipy.spatial.ConvexHull(V)
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        theta = rng.uniform(0, 2 * np.pi, size=k)

        r = np.full(k, radius) if equal_norm else rng.uniform(*radial_range, size=k)
        V = r[:, None] * np.c_[np.cos(theta), np.sin(theta)]
        hull = ConvexHull(V)

        # bijection?
        if enforce_bijection and len(hull.vertices) != k:
            continue

        # self‑inclusion?
        if enforce_self_inclusion:
            ok = True
            for i, vi in enumerate(V):
                if any(vj @ vi > vi @ vi + 1e-14 for j, vj in enumerate(V) if j != i):
                    ok = False
                    break
            if not ok:
                continue

        return V, hull

    raise RuntimeError("Could not satisfy constraints after many tries.")


def compute_cell_map(grid: np.ndarray, basis: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Assign grid points to arg‑max vertices (‑1 for outside)."""
    cell_map = np.argmax(grid @ basis.T, axis=1)
    cell_map[~inside] = -1
    return cell_map


# ------------------------------------------------------------
# Main drawing routine
# ------------------------------------------------------------
def plot_cells(
    k: int = 5,
    *,
    equal_norm: bool = False,
    enforce_bijection: bool = False,
    normalize_inner_product: bool = False,
    enforce_self_inclusion: bool = False,
    save_pdf: bool = True,
    seed: int | None = 42,
) -> None:
    """
    Draw dominance cells for a random k‑gon (with options).

    Labels 𝒞ᵢ(v) are shown only when k ≤ 5.
    """
    # ---------- generate vertices ------------------------------------------
    V, hull = sample_vertices(
        k,
        seed=seed,
        equal_norm=equal_norm,
        enforce_bijection=enforce_bijection,
        enforce_self_inclusion=enforce_self_inclusion,
    )
    K = V[hull.vertices]              # hull vertices in CCW order
    vertex_labels = hull.vertices

    # ---------- dense grid --------------------------------------------------
    pad = 0.1
    xmin, ymin = K.min(axis=0) - pad
    xmax, ymax = K.max(axis=0) + pad
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 1000),
                         np.linspace(ymin, ymax, 1000))
    grid   = np.c_[xx.ravel(), yy.ravel()]
    inside = Path(K).contains_points(grid)

    # ---------- cell assignment --------------------------------------------
    basis = K / np.linalg.norm(K, axis=1, keepdims=True) if normalize_inner_product else K
    cell_map = compute_cell_map(grid, basis, inside)

    # ---------- plot --------------------------------------------------------
    plt.figure(figsize=(8, 8))
    hues   = np.linspace(0, 1, len(K), endpoint=False)
    colors = hsv_to_rgb(np.stack([hues, np.ones(len(K)), np.ones(len(K))], 1))

    show_region_labels = (k <= 5)

    for i in range(len(K)):
        mask = (cell_map == i)
        if not np.any(mask):
            continue
        pts = grid[mask]
        try:
            ch          = ConvexHull(pts)
            hull_coords = pts[ch.vertices]
            plt.gca().add_patch(MplPolygon(hull_coords, facecolor=colors[i],
                                           edgecolor='k', linewidth=0.3))
            # region label if k ≤ 5
            if show_region_labels:
                centroid    = hull_coords.mean(axis=0)
                idx_label   = vertex_labels[i] + 1
                plt.text(*centroid, fr"$\mathscr{{C}}_{{{idx_label}}}(v)$",
                         ha='center', va='center', fontsize=13)
        except Exception:
            pass  # degenerate cell

    # hull outline & vertices
    plt.plot(*K[np.r_[np.arange(len(K)), 0]].T, 'k', lw=1)
    plt.scatter(K[:, 0], K[:, 1], facecolors='none',
                edgecolors='black', s=80, linewidths=1.5)

    # vertex labels
    centroid_K   = K.mean(axis=0)
    offset_scale = 0.05
    for i, v in enumerate(K):
        idx       = vertex_labels[i] + 1
        label_pos = v + offset_scale * (v - centroid_K) / np.linalg.norm(v - centroid_K)
        plt.text(*label_pos, fr"$v_{{{idx}}}$", ha='center', va='center', fontsize=13)

    # origin
    plt.scatter(0, 0, color='black', marker='x', s=80, linewidths=2)
    plt.text(0.02, 0.02, "$0$", fontsize=13)

    plt.axis('equal')
    plt.axis('off')

    # ---------- save / show -------------------------------------------------
    if save_pdf:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tag = "-".join([
            f"k{k}",
            "eq"   if equal_norm            else "var",
            "bij"  if enforce_bijection     else "free",
            "unit" if normalize_inner_product else "raw",
            "self" if enforce_self_inclusion   else "noself",
        ])
        fname = os.path.join(script_dir, f"cells-{tag}.pdf")
        plt.savefig(fname, bbox_inches='tight')
        print(f"Figure saved to {fname}")

    plt.show()


# ------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example: mixed norms, 9 vertices → labels suppressed automatically
    plot_cells(k=6,
               equal_norm=False,
               enforce_bijection=True,
               normalize_inner_product=True)
