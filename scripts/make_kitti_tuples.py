"""Training tuples for NDT-Transformer on KITTI, in the official pickle format.

{i: {"query": relative path, "northing", "easting", "positives": [...],
     "negatives": [...]}} over the training frames [0, split) of one sequence.

The thresholds default to the ones of the BEVPlace++ KITTI training code
(positives < 5 m, negatives > 7 m), so both methods are trained on the same
frames with the same definition of a positive. The original NDT-Transformer
setting on Oxford was 10 m / 50 m (PointNetVLAD protocol).

Example:
  python scripts/make_kitti_tuples.py --kitti_root /content/kitti --seq 00 \
      --ndt_dir /content/ndt --out /content/ndt/train_00.pickle
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
from sklearn.neighbors import KDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr import kitti  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--ndt_dir", required=True, help="dataset folder passed to the trainer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", type=int, default=None, help="default: BEVPlace++ split point")
    ap.add_argument("--pos_thr", type=float, default=5.0)
    ap.add_argument("--neg_thr", type=float, default=7.0)
    args = ap.parse_args()

    split = args.split or kitti.SPLIT_POINTS[args.seq]
    xyz = kitti.positions(kitti.load_poses(args.kitti_root, args.seq))[:split]
    tree = KDTree(xyz)
    pos = tree.query_radius(xyz, r=args.pos_thr)
    near = tree.query_radius(xyz, r=args.neg_thr)
    all_idx = np.arange(split)
    queries = {}
    for i in range(split):
        rel = os.path.join(args.seq, f"{i:06d}.bin")
        assert os.path.exists(os.path.join(args.ndt_dir, rel)), f"missing {rel}"
        p = sorted(int(j) for j in pos[i] if j != i)
        n = np.setdiff1d(all_idx, near[i]).tolist()
        queries[i] = {"query": rel, "northing": float(xyz[i, 0]), "easting": float(xyz[i, 2]),
                      "positives": p, "negatives": n}
    n_pos = np.array([len(q["positives"]) for q in queries.values()])
    with open(args.out, "wb") as f:
        pickle.dump(queries, f, protocol=pickle.HIGHEST_PROTOCOL)
    # sidecar read by the trainer, so the thresholds end up in the run's config.json
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"seq": args.seq, "split": split, "pos_thr": args.pos_thr,
                   "neg_thr": args.neg_thr, "n_queries": split,
                   "median_positives": float(np.median(n_pos))}, f, indent=1)
    print(f"{split} training queries; positives per query: median {np.median(n_pos):.0f}, "
          f"min {n_pos.min()}; queries with <2 positives (skipped by the trainer): "
          f"{(n_pos < 2).sum()} -> {args.out}")


if __name__ == "__main__":
    main()
