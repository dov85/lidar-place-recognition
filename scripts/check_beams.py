"""Check the ring recovery of src/lpr/beams.py on real KITTI scans before building the sparse
variants: rings found per scan and how (scan order or elevation fallback), points kept with
32 and 16 beams, and a picture of one scan with 64 / 32 / 16 beams (top view and elevation
angle against azimuth, where every ring is one horizontal band).

  python scripts/check_beams.py --kitti_root ~/work/kitti --seq 00 --n 30 --out beams_check.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr import kitti  # noqa: E402
from lpr.beams import drop_beams, ring_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--n", type=int, default=30, help="scans checked (spread over the sequence)")
    ap.add_argument("--out", default="beams_check.png")
    a = ap.parse_args()
    total = kitti.num_frames(a.kitti_root, a.seq)
    idxs = np.linspace(0, total - 1, a.n).astype(int)
    rows = []
    for i in idxs:
        scan = kitti.load_scan(kitti.scan_path(a.kitti_root, a.seq, i))
        ring, n, method = ring_index(scan)
        k32 = len(drop_beams(scan, 32)[0])
        k16 = len(drop_beams(scan, 16)[0])
        cnt = np.bincount(ring)
        rows.append((i, n, method, len(scan), k32 / len(scan), k16 / len(scan), cnt.min(), cnt.max()))
        print(f"scan {i:6d}: {n} rings ({method}), points {len(scan)}, kept 32: {k32 / len(scan):.2f}, "
              f"16: {k16 / len(scan):.2f}, points per ring {cnt.min()}-{cnt.max()}")
    by_order = sum(r[2] == "order" for r in rows)
    print(f"\n{by_order}/{len(rows)} scans: rings from the scan order; "
          f"{len(rows) - by_order} from the elevation fallback")
    print(f"kept points: 32 beams {np.mean([r[4] for r in rows]):.3f}, "
          f"16 beams {np.mean([r[5] for r in rows]):.3f} (about 0.5 and 0.25 expected)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scan = kitti.load_scan(kitti.scan_path(a.kitti_root, a.seq, int(idxs[0])))
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    for col, b in enumerate((64, 32, 16)):
        p = drop_beams(scan, b)[0]
        m = (np.abs(p[:, 0]) < 40) & (np.abs(p[:, 1]) < 40)
        axes[0, col].scatter(p[m, 0], p[m, 1], s=0.05, c="k")
        axes[0, col].set_aspect("equal")
        axes[0, col].set_title(f"{b} beams: {len(p)} points (top view, 80 x 80 m)")
        az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        el = np.degrees(np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1])))
        axes[1, col].scatter(az, el, s=0.02, c="k")
        axes[1, col].set_xlabel("azimuth (deg)")
        axes[1, col].set_ylabel("elevation (deg)")
        axes[1, col].set_title(f"{b} beams: one band per ring")
    fig.suptitle(f"KITTI {a.seq}/{int(idxs[0]):06d}")
    fig.tight_layout()
    fig.savefig(a.out, dpi=90)
    print("picture:", a.out)


if __name__ == "__main__":
    main()
