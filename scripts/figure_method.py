"""Method figure of the report: one KITTI scan in the two representations.

  (a) the cropped scan from above, coloured by height;
  (b) the 2000 NDT cells after ground removal (2-sigma ellipses of the x-y marginals);
  (c) the BEVPlace++ density image of the same scan.
Same preprocessing as the experiments (NDT_DEFAULTS, BEV_DEFAULTS).

  python scripts/figure_method.py --kitti_root /content/kitti --seq 08 --idx 30 --out fig_method.png
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from lpr import kitti, ndt_check  # noqa: E402
from lpr.bev import make_bev  # noqa: E402
from lpr.experiments import BEV_DEFAULTS, NDT_DEFAULTS  # noqa: E402
from lpr.ndt_g import ndt_cells  # noqa: E402
from lpr.preprocess import crop, preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--idx", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", type=float, default=40.0, help="half size of the shown area (m)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = NDT_DEFAULTS
    scan = kitti.load_scan(kitti.scan_path(a.kitti_root, a.seq, a.idx))
    cropped = crop(scan, P["crop"], P["half_size"])
    pts = preprocess(scan, P["ground"], P["half_size"], P["crop"])
    means, covs, _ = ndt_cells(pts, **{k: P[k] for k in ("k", "over", "radius", "min_pts", "knn", "cov_eps")})
    bev_args = {k: BEV_DEFAULTS[k] for k in ("half_size", "z_abs", "voxel", "res", "max_count", "ground")
                if k in BEV_DEFAULTS}
    img, _ = make_bev(scan, **bev_args)

    z = a.zoom
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    sc = axes[0].scatter(cropped[:, 0], cropped[:, 1], s=0.05, c=cropped[:, 2], cmap="viridis",
                         vmin=-2.5, vmax=2.0, rasterized=True)
    fig.colorbar(sc, ax=axes[0], fraction=0.046, label="z (m)")
    axes[0].set_title(f"(a) point cloud, {len(cropped):,} points")
    axes[1].scatter(pts[:, 0], pts[:, 1], s=0.02, c="0.75", rasterized=True)
    ndt_check._ellipses(axes[1], means, covs, "tab:red")
    axes[1].set_title(f"(b) {len(means)} NDT cells (ground removed)")
    for ax in axes[:2]:
        ax.set_xlim(-z, z); ax.set_ylim(-z, z); ax.set_aspect("equal")
        ax.set_xlabel("x (m, forward)"); ax.set_ylabel("y (m, left)")
    h = BEV_DEFAULTS["half_size"]
    # image rows = -x, columns = -y (src/lpr/bev.py): transposed and flipped so that, as in
    # (a) and (b), x points right and y up
    axes[2].imshow(img.T[:, ::-1], cmap="gray_r", extent=[-h, h, -h, h])
    axes[2].set_xlim(-z, z); axes[2].set_ylim(-z, z)
    axes[2].set_title(f"(c) BEV density image, {img.shape[0]}x{img.shape[1]} px, 0.4 m/px")
    axes[2].set_xlabel("x (m, forward)"); axes[2].set_ylabel("y (m, left)")
    fig.suptitle(f"KITTI {a.seq}/{a.idx:06d}")
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print("->", a.out)


if __name__ == "__main__":
    main()
