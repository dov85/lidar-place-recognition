"""What happens to the NDT cells when the ground is NOT removed.

For a sample of scans of a sequence, the cells are built twice from the same cropped scan
(the variant parameters of NDT_DEFAULTS, plus --params):
  patchwork  ground removed with Patchwork++ first (the project default)
  none       every cropped point, ground included
Patchwork++ also labels the points of the cropped scan as ground / non-ground; that label is
the reference for both modes (a point is "ground" if Patchwork++ calls it ground).

Columns per scan and mode (coverage = % of the points within --dist m of a cell mean, the
cell radius by default):
  n_points        points the cells were built from
  cov_all         coverage of all cropped points (ground included)
  cov_nonground   coverage of the non-ground points (structures: buildings, cars, trees, ...)
  cov_ground      coverage of the ground points
  cells_ground    % of the cells whose mean is nearest to a ground point (cells spent on ground)
  voxel           voxel size found by the binary search of G (m)
  cells_before    valid cells before the removal of redundant cells; padded: repeated cells
  + the columns of src/lpr/ndt_check.py (cov_0.8, dense_cov, ...) on the points the cells
    were built from, with the prefix pts_
Also: ground_share (% of the cropped points that are ground), and pictures of --previews scans
(cropped scan coloured by the ground label, cells with ground removal, cells without).

Output in --out_dir:
  ground_coverage_<seq>.csv          one row per scan and mode
  ground_coverage_<seq>_summary.csv  mean / 5th percentile per mode
  ground_previews_<seq>/<idx>.png

Example:
  python scripts/ground_coverage.py --kitti_root /content/kitti --seq 00 --n_scans 300 \
      --out_dir /content/drive/MyDrive/lpr/ground_check
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr import kitti, ndt_check  # noqa: E402
from lpr.beams import drop_beams  # noqa: E402
from lpr.experiments import NDT_DEFAULTS  # noqa: E402
from lpr.ndt_g import ndt_cells  # noqa: E402
from lpr.preprocess import _get_patchwork, KITTI_SENSOR_HEIGHT_M, crop  # noqa: E402

MODES = ["patchwork", "none"]
G_KEYS = ["k", "over", "radius", "min_pts", "knn", "cov_eps"]
A = None
P = None


def ground_label(cropped):
    """True for the points Patchwork++ calls ground."""
    pw = _get_patchwork(KITTI_SENSOR_HEIGHT_M)
    pw.estimateGround(np.ascontiguousarray(cropped, dtype=np.float64))
    is_ground = np.ones(len(cropped), dtype=bool)
    is_ground[np.asarray(pw.getNongroundIndices(), dtype=np.int64)] = False
    return is_ground


def pct(mask):
    return round(100.0 * float(np.mean(mask)), 3) if len(mask) else float("nan")


def mode_row(xyz, is_ground, pts, means, info):
    d, _ = cKDTree(means).query(xyz)
    near = d <= A.dist
    _, nearest_pt = cKDTree(xyz).query(means)
    row = {"n_points": len(pts), "cov_all": pct(near), "cov_nonground": pct(near[~is_ground]),
           "cov_ground": pct(near[is_ground]), "cells_ground": pct(is_ground[nearest_pt]),
           "voxel": round(info["voxel"], 4), "cells_before": info["cells_before_removal"],
           "padded": info["padded"]}
    row.update({f"pts_{k}": v for k, v in ndt_check.coverage_metrics(pts, means).items()})
    return row


def preview(path, xyz, is_ground, cells, rows, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = P["half_size"]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax in axes:
        ax.set_xlim(-h, h)
        ax.set_ylim(-h, h)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, forward)")
        ax.set_ylabel("y (m, left)")
    axes[0].scatter(xyz[is_ground, 0], xyz[is_ground, 1], s=0.03, c="tab:olive")
    axes[0].scatter(xyz[~is_ground, 0], xyz[~is_ground, 1], s=0.05, c="k")
    axes[0].set_title(f"cropped scan ({len(xyz)} points)\nolive: ground "
                      f"({pct(is_ground):.1f}%), black: non-ground")
    for ax, mode in zip(axes[1:], MODES):
        means, covs = cells[mode]
        r = rows[mode]
        ax.scatter(xyz[is_ground, 0], xyz[is_ground, 1], s=0.02, c="#d8dcc0")
        ax.scatter(xyz[~is_ground, 0], xyz[~is_ground, 1], s=0.02, c="0.6")
        ndt_check._ellipses(ax, means, covs, "tab:red")
        name = "ground removed (Patchwork++)" if mode == "patchwork" else "ground kept"
        ax.set_title(f"{name}: {len(means)} cells, {r['cells_ground']:.0f}% on ground\n"
                     f"covered: all {r['cov_all']:.1f}%, non-ground {r['cov_nonground']:.1f}%, "
                     f"ground {r['cov_ground']:.1f}%")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def check(idx):
    scan = kitti.load_scan(kitti.scan_path(A.kitti_root, A.seq, idx))
    scan, _ = drop_beams(scan, P["beams"])
    cropped = crop(scan, P["crop"], P["half_size"])
    xyz = cropped[:, :3].astype(np.float64)
    is_ground = ground_label(cropped)
    g = {k: P[k] for k in G_KEYS}
    out, cells, rows = [], {}, {}
    for mode in MODES:
        pts = xyz[~is_ground] if mode == "patchwork" else xyz
        t0 = time.time()
        means, covs, info = ndt_cells(pts, **g)
        rows[mode] = {"idx": idx, "mode": mode, "ground_share": pct(is_ground),
                      **mode_row(xyz, is_ground, pts, means, info),
                      "g_seconds": round(time.time() - t0, 2)}
        cells[mode] = (means, covs)
        out.append(rows[mode])
    if idx in A.preview_idx:
        preview(os.path.join(A.out_dir, f"ground_previews_{A.seq}", f"{idx:06d}.png"),
                xyz, is_ground, cells, rows, f"scan {A.seq}/{idx:06d}")
    return out


def main():
    global A, P
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_scans", type=int, default=300, help="random scans (0 = all)")
    ap.add_argument("--previews", type=int, default=6, help="scans with a picture")
    ap.add_argument("--dist", type=float, default=None, help="coverage distance (default: radius)")
    ap.add_argument("--params", default="{}", help="JSON: NDT parameters that differ from the defaults")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    A = ap.parse_args()
    P = {**NDT_DEFAULTS, **json.loads(A.params)}
    A.dist = A.dist if A.dist is not None else P["radius"]
    n = kitti.num_frames(A.kitti_root, A.seq)
    rng = np.random.default_rng([A.seed, int(A.seq)])
    idxs = (list(range(n)) if A.n_scans <= 0 or A.n_scans >= n
            else sorted(int(i) for i in rng.choice(n, A.n_scans, replace=False)))
    A.preview_idx = set(idxs[:: max(1, len(idxs) // max(A.previews, 1))][: A.previews])
    os.makedirs(os.path.join(A.out_dir, f"ground_previews_{A.seq}"), exist_ok=True)
    print(f"seq {A.seq}: {len(idxs)} of {n} scans, modes {MODES}, coverage within {A.dist} m, "
          f"{A.workers} workers", flush=True)

    t0 = time.time()
    rows = []
    with Pool(A.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(check, idxs, chunksize=4), 1):
            rows += r
            if i % 50 == 0 or i == len(idxs):
                el = time.time() - t0
                print(f"  {i}/{len(idxs)} scans, {el / 60:.1f} min, "
                      f"~{el / i * (len(idxs) - i) / 60:.1f} min left", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows).sort_values(["idx", "mode"])
    df.insert(0, "seq", A.seq)
    df.to_csv(os.path.join(A.out_dir, f"ground_coverage_{A.seq}.csv"), index=False)
    cols = ["n_points", "cov_all", "cov_nonground", "cov_ground", "cells_ground", "voxel",
            "cells_before", "padded", "pts_cov_0.8", "pts_dense_cov", "ground_share", "g_seconds"]
    summ = df.groupby("mode")[cols].mean().round(2)
    p5 = df.groupby("mode")[["cov_all", "cov_nonground"]].quantile(0.05).round(2)
    summ = summ.join(p5.add_suffix(" p5")).reindex(MODES)
    summ.insert(0, "scans", df.groupby("mode").size())
    summ.insert(0, "seq", A.seq)
    summ.to_csv(os.path.join(A.out_dir, f"ground_coverage_{A.seq}_summary.csv"))
    with open(os.path.join(A.out_dir, f"ground_coverage_{A.seq}_params.json"), "w") as f:
        json.dump({"seq": A.seq, "scans": idxs, "dist": A.dist, "params": P,
                   "previews": sorted(A.preview_idx)}, f, indent=1)
    print(summ.T.to_string())


if __name__ == "__main__":
    main()
