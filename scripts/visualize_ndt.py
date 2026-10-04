"""Top-down picture of one scan: raw points, points left after ground removal and
the 2000 NDT cells drawn as 2-sigma ellipses (x-y marginal of each Gaussian).

Example:
  python scripts/visualize_ndt.py --kitti_root /content/kitti --seq 00 --idx 0 --out cells.png
"""
import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import EllipseCollection  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr import kitti  # noqa: E402
from lpr.ndt_g import ndt_cells  # noqa: E402
from lpr.preprocess import crop_box, preprocess  # noqa: E402


def ellipses(ax, means, covs, color):
    c2 = covs[:, :2, :2]
    vals, vecs = np.linalg.eigh(c2)
    vals = np.clip(vals, 1e-6, None)
    w, h = 2 * 2 * np.sqrt(vals[:, 1]), 2 * 2 * np.sqrt(vals[:, 0])
    ang = np.degrees(np.arctan2(vecs[:, 1, 1], vecs[:, 0, 1]))
    ax.add_collection(EllipseCollection(w, h, ang, units="xy", offsets=means[:, :2],
                                        offset_transform=ax.transData, facecolors="none",
                                        edgecolors=color, linewidths=0.6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--ground", default="patchwork")
    ap.add_argument("--out", default="ndt_cells.png")
    args = ap.parse_args()

    scan = kitti.load_scan(kitti.scan_path(args.kitti_root, args.seq, args.idx))
    t0 = time.time()
    pts = preprocess(scan, ground=args.ground)
    t1 = time.time()
    means, covs, info = ndt_cells(pts)
    t2 = time.time()
    raw = crop_box(scan)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax in axes:
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_aspect("equal")
    axes[0].scatter(raw[:, 0], raw[:, 1], s=0.05, c="k")
    axes[0].set_title(f"cropped scan ({len(raw)} points)")
    axes[1].scatter(pts[:, 0], pts[:, 1], s=0.05, c="k")
    axes[1].set_title(f"after ground removal ({len(pts)} points, {t1 - t0:.2f} s)")
    axes[2].scatter(pts[:, 0], pts[:, 1], s=0.02, c="0.7")
    ellipses(axes[2], means, covs, "tab:red")
    axes[2].set_title(f"{len(means)} NDT cells (voxel {info['voxel']:.2f} m, "
                      f"{info['removed']} removed, {t2 - t1:.2f} s)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(info, f"preprocess {t1 - t0:.2f}s, G {t2 - t1:.2f}s ->", args.out)


if __name__ == "__main__":
    main()
