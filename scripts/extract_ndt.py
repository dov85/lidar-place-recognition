"""Run preprocessing + G on every scan of a KITTI sequence.

Output: <out_dir>/<seq>/<idx:06d>.bin in the official NDT-Transformer format
(2000 x 12 float64), plus <out_dir>/<seq>_stats.csv with one row per scan.
Existing outputs are skipped, so the script can be resumed.

Example:
  python scripts/extract_ndt.py --kitti_root /content/kitti --seq 00 --out_dir /content/ndt
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr import kitti  # noqa: E402
from lpr.ndt_g import DEFAULTS, ndt_cells, to_official_format  # noqa: E402
from lpr.preprocess import preprocess  # noqa: E402

ARGS = None


def process(idx):
    out = os.path.join(ARGS.out_dir, ARGS.seq, f"{idx:06d}.bin")
    if os.path.exists(out) and not ARGS.overwrite:
        return None
    t0 = time.time()
    scan = kitti.load_scan(kitti.scan_path(ARGS.kitti_root, ARGS.seq, idx))
    pts = preprocess(scan, ground=ARGS.ground, half_size=ARGS.half_size)
    means, covs, info = ndt_cells(pts, k=ARGS.k, over=ARGS.over, radius=ARGS.radius,
                                  min_pts=ARGS.min_pts)
    arr = to_official_format(means, covs, scale=ARGS.half_size)
    tmp = out + ".tmp"
    arr.tofile(tmp)
    os.replace(tmp, out)
    info.update(idx=idx, n_raw=len(scan), seconds=round(time.time() - t0, 3))
    return info


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ground", default="patchwork", choices=["patchwork", "ransac", "none"])
    ap.add_argument("--half_size", type=float, default=40.0)
    ap.add_argument("--k", type=int, default=DEFAULTS["k"])
    ap.add_argument("--over", type=float, default=DEFAULTS["over"])
    ap.add_argument("--radius", type=float, default=DEFAULTS["radius"])
    ap.add_argument("--min_pts", type=int, default=DEFAULTS["min_pts"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--first", type=int, default=None, help="only the first N scans (testing)")
    ap.add_argument("--overwrite", action="store_true")
    ARGS = ap.parse_args()

    os.makedirs(os.path.join(ARGS.out_dir, ARGS.seq), exist_ok=True)
    n = kitti.num_frames(ARGS.kitti_root, ARGS.seq)
    idxs = list(range(n if ARGS.first is None else min(n, ARGS.first)))
    stats_path = os.path.join(ARGS.out_dir, f"{ARGS.seq}_stats.csv")
    new_file = not os.path.exists(stats_path)
    fields = ["idx", "n_raw", "n_points", "voxel", "cells_before_removal", "removed",
              "padded", "seconds"]
    t0 = time.time()
    done = 0
    with open(stats_path, "a", newline="") as f, Pool(ARGS.workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for info in pool.imap_unordered(process, idxs, chunksize=4):
            done += 1
            if info is not None:
                w.writerow({k: info[k] for k in fields})
            if done % 200 == 0 or done == len(idxs):
                el = time.time() - t0
                print(f"{done}/{len(idxs)} scans, {el / 60:.1f} min elapsed, "
                      f"~{el / done * (len(idxs) - done) / 60:.1f} min left", flush=True)
                f.flush()
    print("done:", os.path.join(ARGS.out_dir, ARGS.seq))


if __name__ == "__main__":
    main()
