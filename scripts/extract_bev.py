"""BEV density image (BEVPlace++) of every scan of a KITTI sequence.

Output: <out_dir>/<seq>/<idx:06d>.png (201 x 201 grey), <out_dir>/<seq>_stats.csv with
one row per scan, and <out_dir>/config.json with every parameter (the variant documents
itself, like extract_ndt.py). Existing images are skipped, so the script can be resumed.

Example:
  python scripts/extract_bev.py --kitti_root /content/kitti --seq 00 --out_dir /content/bev/<variant>
"""
import argparse
import csv
import datetime
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr import kitti  # noqa: E402
from lpr.bev import make_bev  # noqa: E402
from lpr.experiments import BEV_DEFAULTS, full_params, git_commit, ndt_hash, write_json  # noqa: E402

ARGS = None
PARAMS = None


def process(idx):
    out = os.path.join(ARGS.out_dir, ARGS.seq, f"{idx:06d}.png")
    if os.path.exists(out) and not ARGS.overwrite:
        return None
    t0 = time.time()
    scan = kitti.load_scan(kitti.scan_path(ARGS.kitti_root, ARGS.seq, idx))
    img, info = make_bev(scan, **PARAMS)
    tmp = os.path.join(ARGS.out_dir, ARGS.seq, f".{idx:06d}.tmp.png")
    cv2.imwrite(tmp, img)
    os.replace(tmp, out)
    info.update(idx=idx, seconds=round(time.time() - t0, 3))
    return info


def main():
    global ARGS, PARAMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--out_dir", required=True)
    d = BEV_DEFAULTS
    ap.add_argument("--half_size", type=float, default=d["half_size"])
    ap.add_argument("--z_abs", type=float, default=d["z_abs"])
    ap.add_argument("--voxel", type=float, default=d["voxel"])
    ap.add_argument("--res", type=float, default=d["res"])
    ap.add_argument("--max_count", type=int, default=d["max_count"])
    ap.add_argument("--ground", default=d["ground"], choices=["none", "patchwork", "ransac"])
    ap.add_argument("--variant", default="", help="name recorded in config.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--first", type=int, default=None, help="only the first N scans (testing)")
    ap.add_argument("--overwrite", action="store_true")
    ARGS = ap.parse_args()
    PARAMS = full_params(BEV_DEFAULTS, {k: getattr(ARGS, k) for k in BEV_DEFAULTS})
    phash = ndt_hash(PARAMS)  # same hash function for every method

    os.makedirs(os.path.join(ARGS.out_dir, ARGS.seq), exist_ok=True)
    cfg_path = os.path.join(ARGS.out_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
    if cfg is not None and cfg["params_hash"] != phash:
        raise SystemExit(f"{ARGS.out_dir} holds a variant with other parameters "
                         f"({cfg['params_hash']} != {phash}); use another out_dir")
    started = datetime.datetime.now().isoformat(timespec="seconds")
    n = kitti.num_frames(ARGS.kitti_root, ARGS.seq)
    idxs = list(range(n if ARGS.first is None else min(n, ARGS.first)))
    stats_path = os.path.join(ARGS.out_dir, f"{ARGS.seq}_stats.csv")
    new_file = not os.path.exists(stats_path)
    fields = ["idx", "n_raw", "n_points", "n_voxels", "occupied_px", "seconds"]
    t0 = time.time()
    done = 0
    with open(stats_path, "a", newline="") as f, Pool(ARGS.workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for info in pool.imap_unordered(process, idxs, chunksize=8):
            done += 1
            if info is not None:
                w.writerow({k: info[k] for k in fields})
            if done % 500 == 0 or done == len(idxs):
                el = time.time() - t0
                print(f"{done}/{len(idxs)} scans, {el / 60:.1f} min elapsed, "
                      f"~{el / done * (len(idxs) - done) / 60:.1f} min left", flush=True)
                f.flush()
    import pandas as pd
    st = pd.read_csv(stats_path)
    cfg = cfg or {"method": "bev", "variant": ARGS.variant, "params": PARAMS, "params_hash": phash,
                  "seqs": {}}
    cfg.setdefault("seqs", {})[ARGS.seq] = {
        "code_commit": git_commit(os.path.join(os.path.dirname(__file__), "..")),
        "started": started,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_scans": int(len(st)),
        "stats_mean": st[fields[1:]].mean().round(4).to_dict(),
    }
    write_json(cfg_path, cfg)
    print("done:", os.path.join(ARGS.out_dir, ARGS.seq), "| config:", cfg_path)


if __name__ == "__main__":
    main()
