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
from lpr.experiments import NDT_DEFAULTS  # noqa: E402
from lpr.ndt_g import ndt_cells  # noqa: E402
from lpr.beams import drop_beams  # noqa: E402
from lpr.preprocess import crop, preprocess  # noqa: E402


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
    ap.add_argument("--out", default="ndt_cells.png")
    d = NDT_DEFAULTS  # same options as extract_ndt.py, so a preview matches its variant
    ap.add_argument("--ground", default=d["ground"])
    ap.add_argument("--crop", default=d["crop"])
    ap.add_argument("--half_size", type=float, default=d["half_size"])
    ap.add_argument("--k", type=int, default=d["k"])
    ap.add_argument("--over", type=float, default=d["over"])
    ap.add_argument("--radius", type=float, default=d["radius"])
    ap.add_argument("--min_pts", type=int, default=d["min_pts"])
    ap.add_argument("--knn", type=int, default=d["knn"])
    ap.add_argument("--cov_eps", type=float, default=d["cov_eps"])
    ap.add_argument("--beams", type=int, default=d["beams"], choices=[64, 32, 16])
    ap.add_argument("--variant", default="")
    args = ap.parse_args()

    scan = kitti.load_scan(kitti.scan_path(args.kitti_root, args.seq, args.idx))
    scan, _ = drop_beams(scan, args.beams)
    t0 = time.time()
    pts = preprocess(scan, ground=args.ground, half_size=args.half_size, crop_shape=args.crop)
    t1 = time.time()
    means, covs, info = ndt_cells(pts, k=args.k, over=args.over, radius=args.radius,
                                  min_pts=args.min_pts, knn=args.knn, cov_eps=args.cov_eps)
    t2 = time.time()
    raw = crop(scan, args.crop, args.half_size)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax in axes:
        ax.set_xlim(-args.half_size, args.half_size)
        ax.set_ylim(-args.half_size, args.half_size)
        ax.set_aspect("equal")
    axes[0].scatter(raw[:, 0], raw[:, 1], s=0.05, c="k")
    axes[0].set_title(f"cropped scan ({len(raw)} points)")
    axes[1].scatter(pts[:, 0], pts[:, 1], s=0.05, c="k")
    axes[1].set_title(f"after ground removal ({len(pts)} points, {t1 - t0:.2f} s)")
    axes[2].scatter(pts[:, 0], pts[:, 1], s=0.02, c="0.7")
    ellipses(axes[2], means, covs, "tab:red")
    axes[2].set_title(f"{len(means)} NDT cells (voxel {info['voxel']:.2f} m, "
                      f"{info['removed']} removed, {t2 - t1:.2f} s)")
    if args.variant:
        fig.suptitle(f"{args.variant}  |  scan {args.seq}/{args.idx:06d}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(info, f"preprocess {t1 - t0:.2f}s, G {t2 - t1:.2f}s ->", args.out)


if __name__ == "__main__":
    main()
