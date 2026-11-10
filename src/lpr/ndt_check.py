"""Quality checks of NDT cells, run by every extraction (scripts/extract_ndt.py).

Coverage: how well the cells describe the points they were built from (the points left
after crop and ground removal). Columns per scan (all distances in metres):
  n_points       points after crop and ground removal
  cov_0.8        % of the points within 0.8 m (the cell radius) of a cell mean
  cov_1.5        % of the points within 1.5 m of a cell mean
  median_dist    median distance from a point to the nearest cell mean
  dense_cov      % of the 1 m x 1 m squares with >= 5 points that have a cell mean within 1 m
                 of their centre (should be ~100: every dense structure has a cell)
  sparse_share   % of the points in squares with < 5 points (isolated points; cells need >= 5
                 points within 0.8 m, so these are not expected to be covered)
  cov_0-10 ...   cov_0.8 per range from the sensor: 0-10, 10-20, 20-30, >= 30 m
  cells_x_min ... extent of the cell means
Previews: for a few scans per sequence, a top view of the cropped scan, the points after
ground removal and the cells (2-sigma ellipses of the x-y marginals).
"""
import numpy as np
from scipy.spatial import cKDTree

RANGES = [(0, 10), (10, 20), (20, 30), (30, np.inf)]
COVERAGE_FIELDS = (["cov_0.8", "cov_1.5", "median_dist", "dense_cov", "sparse_share"]
                   + [f"cov_{lo}-{hi}" if np.isfinite(hi) else f"cov_{lo}+" for lo, hi in RANGES]
                   + ["cells_x_min", "cells_x_max", "cells_y_min", "cells_y_max"])
N_PREVIEWS = 10


def coverage_metrics(points, means, square=1.0, dense_min=5):
    """points (N, 3) after crop and ground removal, means (k, 3); both in metres."""
    out = {f: float("nan") for f in COVERAGE_FIELDS}
    if len(points) == 0 or len(means) == 0:
        return out
    d, _ = cKDTree(means).query(points)
    out["cov_0.8"] = 100.0 * np.mean(d <= 0.8)
    out["cov_1.5"] = 100.0 * np.mean(d <= 1.5)
    out["median_dist"] = float(np.median(d))
    keys = np.floor(points[:, :2] / square).astype(np.int64)
    sq, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    dense = cnt >= dense_min
    out["sparse_share"] = 100.0 * np.mean(~dense[inv])
    if dense.any():
        dc, _ = cKDTree(means[:, :2]).query((sq[dense] + 0.5) * square)
        out["dense_cov"] = 100.0 * np.mean(dc <= square)
    r = np.hypot(points[:, 0], points[:, 1])
    for (lo, hi), name in zip(RANGES, COVERAGE_FIELDS[5:9]):
        m = (r >= lo) & (r < hi)
        if m.any():
            out[name] = 100.0 * np.mean(d[m] <= 0.8)
    out["cells_x_min"], out["cells_y_min"] = (float(v) for v in means[:, :2].min(axis=0))
    out["cells_x_max"], out["cells_y_max"] = (float(v) for v in means[:, :2].max(axis=0))
    return {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v)
            for k, v in out.items()}


def preview_indices(seq, n_scans, n=N_PREVIEWS, seed=0):
    """The same random scans of a sequence for every variant (seeded by the sequence)."""
    rng = np.random.default_rng([seed, int(seq)])
    return sorted(int(i) for i in rng.choice(n_scans, min(n, n_scans), replace=False))


def _ellipses(ax, means, covs, color):
    from matplotlib.collections import EllipseCollection
    vals, vecs = np.linalg.eigh(covs[:, :2, :2])
    vals = np.clip(vals, 1e-6, None)
    w, h = 4 * np.sqrt(vals[:, 1]), 4 * np.sqrt(vals[:, 0])      # 2-sigma, full axes
    ang = np.degrees(np.arctan2(vecs[:, 1, 1], vecs[:, 0, 1]))
    ax.add_collection(EllipseCollection(w, h, ang, units="xy", offsets=means[:, :2],
                                        offset_transform=ax.transData, facecolors="none",
                                        edgecolors=color, linewidths=0.6))


def save_preview(path, cropped, points, means, covs, half_size, title="", metrics=None):
    """Three top views: cropped scan (before ground removal), after ground removal, cells."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax in axes:
        ax.set_xlim(-half_size, half_size)
        ax.set_ylim(-half_size, half_size)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, forward)")
        ax.set_ylabel("y (m, left)")
    sc = axes[0].scatter(cropped[:, 0], cropped[:, 1], s=0.05, c=cropped[:, 2], cmap="viridis",
                         vmin=-2.5, vmax=2.0)
    fig.colorbar(sc, ax=axes[0], fraction=0.046, label="z (m)")
    axes[0].set_title(f"before ground removal ({len(cropped)} points)")
    axes[1].scatter(cropped[:, 0], cropped[:, 1], s=0.03, c="0.85")
    axes[1].scatter(points[:, 0], points[:, 1], s=0.05, c="k")
    axes[1].set_title(f"after ground removal ({len(points)} points; grey: removed)")
    axes[2].scatter(points[:, 0], points[:, 1], s=0.02, c="0.7")
    _ellipses(axes[2], means, covs, "tab:red")
    t = f"{len(means)} NDT cells (2-sigma)"
    if metrics:
        t += (f"\npoints within 0.8 m of a cell: {metrics['cov_0.8']:.1f}%, "
              f"dense squares covered: {metrics['dense_cov']:.1f}%")
    axes[2].set_title(t)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def summarize(stats):
    """One row per sequence from a stats / coverage table (pandas DataFrame)."""
    stats = stats.copy()
    cols = [c for c in COVERAGE_FIELDS[:9] if c in stats]
    for c in cols:
        stats[c] = stats[c].apply(lambda v: v if v != "" else None).astype(float)
    stats = stats.dropna(subset=cols[:1]) if cols else stats
    row = {"scans": int(len(stats))}
    if not len(stats):
        return row
    for c in cols:
        row[f"{c} mean"] = round(float(stats[c].mean()), 2)
    for c in ("cov_0.8", "dense_cov"):
        if c in stats:
            row[f"{c} p5"] = round(float(stats[c].quantile(0.05)), 2)
            row[f"{c} min"] = round(float(stats[c].min()), 2)
    if "cov_0.8" in stats and "idx" in stats:
        worst = stats.nsmallest(5, "cov_0.8")
        row["worst scans (cov_0.8)"] = " ".join(f"{int(i)}:{v:.1f}"
                                                for i, v in zip(worst["idx"], worst["cov_0.8"]))
    return row
