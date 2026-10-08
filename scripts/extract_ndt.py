"""Run preprocessing + G on every scan of a KITTI sequence.

Output: <out_dir>/<seq>/<idx:06d>.bin in the official NDT-Transformer format
(2000 x 12 float64), <out_dir>/<seq>_stats.csv with one row per scan, and
<out_dir>/config.json with every parameter used (so a variant documents itself).
Existing outputs are skipped, so the script can be resumed.

With --rot_copies N (training data for rot_points): instead of the cells above, N copies of
every scan whose raw points are first rotated about z by a random angle (uniform 0-360 deg,
from --rot_seed, the sequence, the scan and the copy), then cropped and turned into cells
exactly like the unrotated scans (same variant parameters):
  <out_dir>/<seq>_rot<N>/<idx:06d>_<copy>.bin, <out_dir>/<seq>_rot<N>_stats.csv (with the angles)
and config.json gets an entry "rot_copies" -> "<seq>_rot<N>". Only the training frames
(before the BEVPlace++ split point, e.g. 0-2999 of 00) unless --first is given.

Example:
  python scripts/extract_ndt.py --kitti_root /content/kitti --seq 00 --out_dir /content/ndt
"""
import argparse
import csv
import datetime
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr import kitti  # noqa: E402
import json  # noqa: E402

from lpr.experiments import (NDT_DEFAULTS, full_params, git_commit, ndt_hash,  # noqa: E402
                             upgrade_ndt_config, write_json)
from lpr.experiments import rot_dir_name  # noqa: E402
from lpr.ndt_g import ndt_cells, to_official_format  # noqa: E402
from lpr.preprocess import preprocess  # noqa: E402
from lpr.rotation import rz  # noqa: E402

ARGS = None


def rot_angle(seed, seq, idx, copy):
    """Angle (deg) of one point-rotated copy; does not depend on the number of copies."""
    return float(np.random.default_rng([seed, int(seq), idx, copy]).uniform(0.0, 360.0))


def process(item):
    idx, copy = item if isinstance(item, tuple) else (item, None)
    if copy is None:
        out = os.path.join(ARGS.out_dir, ARGS.seq, f"{idx:06d}.bin")
    else:
        out = os.path.join(ARGS.out_dir, rot_dir_name(ARGS.seq, ARGS.rot_copies),
                           f"{idx:06d}_{copy}.bin")
    if os.path.exists(out) and not ARGS.overwrite:
        return None
    t0 = time.time()
    scan = kitti.load_scan(kitti.scan_path(ARGS.kitti_root, ARGS.seq, idx))
    angle = None
    if copy is not None:   # rotate the raw points before crop, ground removal and cells
        angle = rot_angle(ARGS.rot_seed, ARGS.seq, idx, copy)
        scan = scan.copy()
        scan[:, :3] = scan[:, :3] @ rz(angle).T.astype(np.float32)
    pts = preprocess(scan, ground=ARGS.ground, half_size=ARGS.half_size, crop_shape=ARGS.crop)
    means, covs, info = ndt_cells(pts, k=ARGS.k, over=ARGS.over, radius=ARGS.radius,
                                  min_pts=ARGS.min_pts, knn=ARGS.knn, cov_eps=ARGS.cov_eps)
    arr = to_official_format(means, covs, scale=ARGS.half_size)
    tmp = out + ".tmp"
    arr.tofile(tmp)
    os.replace(tmp, out)
    info.update(idx=idx, n_raw=len(scan), seconds=round(time.time() - t0, 3))
    if copy is not None:
        info.update(copy=copy, angle_deg=round(angle, 4))
    return info


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--out_dir", required=True)
    d = NDT_DEFAULTS
    ap.add_argument("--ground", default=d["ground"], choices=["patchwork", "ransac", "none"])
    ap.add_argument("--crop", default=d["crop"], choices=["square", "circle"])
    ap.add_argument("--half_size", type=float, default=d["half_size"],
                    help="crop half-size / radius in m, also the normalisation scale")
    ap.add_argument("--k", type=int, default=d["k"])
    ap.add_argument("--over", type=float, default=d["over"])
    ap.add_argument("--radius", type=float, default=d["radius"])
    ap.add_argument("--min_pts", type=int, default=d["min_pts"])
    ap.add_argument("--knn", type=int, default=d["knn"])
    ap.add_argument("--cov_eps", type=float, default=d["cov_eps"])
    ap.add_argument("--variant", default="", help="name recorded in config.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--first", type=int, default=None, help="only the first N scans (testing)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--rot_copies", type=int, default=0,
                    help="N > 0: build N copies of every scan with the raw points rotated by a "
                         "random angle (training data for rot_points), instead of the cells")
    ap.add_argument("--rot_seed", type=int, default=0)
    # with --rot_copies and no --first: only the training frames (before the split point)
    ARGS = ap.parse_args()

    rot = ARGS.rot_copies > 0
    sub = rot_dir_name(ARGS.seq, ARGS.rot_copies) if rot else ARGS.seq
    os.makedirs(os.path.join(ARGS.out_dir, sub), exist_ok=True)
    params = full_params(NDT_DEFAULTS, {k: getattr(ARGS, k) for k in NDT_DEFAULTS})
    cfg_path = os.path.join(ARGS.out_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
    if cfg is not None:
        cfg, _ = upgrade_ndt_config(cfg)
        if cfg["params_hash"] != ndt_hash(params):
            raise SystemExit(f"{ARGS.out_dir} holds a variant with other parameters "
                             f"({cfg['params_hash']} != {ndt_hash(params)}); use another out_dir")
    started = datetime.datetime.now().isoformat(timespec="seconds")
    n = kitti.num_frames(ARGS.kitti_root, ARGS.seq)
    if rot and ARGS.first is None:   # copies are training data: the training frames only
        ARGS.first = kitti.SPLIT_POINTS[ARGS.seq]
    idxs = list(range(n if ARGS.first is None else min(n, ARGS.first)))
    if rot:
        idxs = [(i, c) for i in idxs for c in range(ARGS.rot_copies)]
    stats_path = os.path.join(ARGS.out_dir, f"{sub}_stats.csv")
    new_file = not os.path.exists(stats_path)
    fields = ["idx", "n_raw", "n_points", "voxel", "cells_before_removal", "removed",
              "padded", "seconds"] + (["copy", "angle_deg"] if rot else [])
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
    import pandas as pd
    st = pd.read_csv(stats_path)
    cfg = cfg or {"variant": ARGS.variant, "params": params, "params_hash": ndt_hash(params),
                  "seqs": {}}
    entry = cfg.setdefault("rot_copies", {}) if rot else cfg.setdefault("seqs", {})
    entry[sub] = {
        "code_commit": git_commit(os.path.join(os.path.dirname(__file__), "..")),
        "started": started,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_scans": int(len(st)),
        "stats_mean": st[["n_raw", "n_points", "voxel", "cells_before_removal", "removed",
                          "padded", "seconds"]].mean().round(4).to_dict(),
        "padded_scans": int((st["padded"] > 0).sum()),
    }
    if rot:
        entry[sub].update(seq=ARGS.seq, copies=ARGS.rot_copies, rot_seed=ARGS.rot_seed,
                          n_files=int(len(st)), n_scans=int(st["idx"].nunique()))
    write_json(cfg_path, cfg)
    print("done:", os.path.join(ARGS.out_dir, sub), "| config:", cfg_path)


if __name__ == "__main__":
    main()
