"""Checks of NDT cells that were built before the checks were part of extract_ndt.py.

For every scan (or every --every-th) of a sequence: the points after crop and ground removal
are recomputed from the raw scan with the variant's parameters (its config.json), the cells
are read from <ndt_dir>/<seq>/<idx>.bin, and the coverage columns of src/lpr/ndt_check.py are
computed. Pictures (before / after ground removal / cells) of the same 10 random scans that
extract_ndt.py draws.

Note: Patchwork++ adapts its thresholds from scan to scan inside a process, so the recomputed
points can differ slightly (typically < 1% of the points) from the ones the cells were built
from; the coverage is therefore exact up to that. extract_ndt.py computes it on the exact points.

Output in --out_dir:
  coverage_<seq>.csv          one row per scan
  coverage_<seq>_summary.json  summary (means, 5th percentile, worst scans)
  previews_<seq>/<idx>.png

Example (on the Azure VM, see scripts/azure_coverage.sh):
  python scripts/ndt_coverage.py --kitti_root ~/work/kitti --ndt_dir ~/work/ndt/<variant> \
      --seq 08 --out_dir ~/work/coverage/<variant>
"""
import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr import kitti, ndt_check  # noqa: E402
from lpr.ndt_g import from_official_format  # noqa: E402
from lpr.preprocess import crop, preprocess  # noqa: E402

A = None
P = None


def check(idx):
    scan = kitti.load_scan(kitti.scan_path(A.kitti_root, A.seq, idx))
    pts = preprocess(scan, ground=P["ground"], half_size=P["half_size"], crop_shape=P["crop"])
    arr = np.fromfile(os.path.join(A.ndt_dir, A.seq, f"{idx:06d}.bin"), dtype=np.float64)
    means, covs = from_official_format(arr, scale=P["half_size"])
    row = {"idx": idx, "n_points": len(pts), **ndt_check.coverage_metrics(pts, means)}
    if idx in A.preview_idx:
        ndt_check.save_preview(os.path.join(A.out_dir, f"previews_{A.seq}", f"{idx:06d}.png"),
                               crop(scan, P["crop"], P["half_size"]), pts, means, covs,
                               P["half_size"], title=f"{A.variant}  |  scan {A.seq}/{idx:06d}",
                               metrics=row)
    return row


def main():
    global A, P
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--ndt_dir", required=True, help="variant folder: config.json, <seq>/*.bin")
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--every", type=int, default=1, help="check every n-th scan")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    A = ap.parse_args()
    cfg = json.load(open(os.path.join(A.ndt_dir, "config.json")))
    P = cfg["params"]
    A.variant = cfg.get("variant", os.path.basename(os.path.normpath(A.ndt_dir)))
    n = len([f for f in os.listdir(os.path.join(A.ndt_dir, A.seq)) if f.endswith(".bin")])
    A.preview_idx = set(ndt_check.preview_indices(A.seq, n))
    idxs = sorted(set(range(0, n, A.every)) | A.preview_idx)
    os.makedirs(os.path.join(A.out_dir, f"previews_{A.seq}"), exist_ok=True)
    print(f"{A.variant} seq {A.seq}: checking {len(idxs)} of {n} scans, {A.workers} workers",
          flush=True)

    t0 = time.time()
    rows = []
    with Pool(A.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(check, idxs, chunksize=8), 1):
            rows.append(row)
            if i % 500 == 0 or i == len(idxs):
                el = time.time() - t0
                print(f"  {i}/{len(idxs)} scans, {el / 60:.1f} min, "
                      f"~{el / i * (len(idxs) - i) / 60:.1f} min left", flush=True)
    rows.sort(key=lambda r: r["idx"])
    path = os.path.join(A.out_dir, f"coverage_{A.seq}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "n_points"] + ndt_check.COVERAGE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    import pandas as pd
    summary = {"variant": A.variant, "seq": A.seq, "scans_in_seq": n, "every": A.every,
               "previews": sorted(A.preview_idx),
               **ndt_check.summarize(pd.read_csv(path))}
    with open(os.path.join(A.out_dir, f"coverage_{A.seq}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
