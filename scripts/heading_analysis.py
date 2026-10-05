"""How different is the heading of each query from the heading of the map frames
at the same place? Needs only the poses; optionally relates it to retrieval success.

For every query (BEVPlace++ split: database = frames before the split point,
queries from 200 frames after it) that has database frames within 5 m:
  - diff_nearest: heading difference to the database frame closest in position;
  - diff_min:     smallest heading difference over all database frames within 5 m
                  (does the map contain the place seen from a similar direction at all?).
Heading = yaw of the camera's forward axis in the ground plane of the KITTI poses.
Differences are in [0, 180] degrees and grouped into
  same (< 45), crossing (45-135) and opposite (> 135).

With --matches_dir (folder with eval_<seq>_matches.npz written by eval_ndt_kitti.py),
the Recall@1 of each heading group is reported as well.

Example:
  python scripts/heading_analysis.py --kitti_root /content/kitti --seqs 00 02 05 06 08 \
      --out_dir /content/drive/MyDrive/lpr/analysis/heading \
      --matches_dir /content/drive/MyDrive/lpr/runs/<run>
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr import kitti  # noqa: E402

GROUPS = [("same (<45)", 0, 45), ("crossing (45-135)", 45, 135), ("opposite (>135)", 135, 180.01)]


def yaw_deg(poses):
    """Heading in degrees: camera forward axis (3rd column of R) in the x-z ground plane."""
    fwd = poses[:, :, 2]
    return np.degrees(np.arctan2(fwd[:, 0], fwd[:, 2]))


def ang_diff(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def analyse(kitti_root, seq, thr, matches_dir=None):
    poses = kitti.load_poses(kitti_root, seq)
    xyz, yaw = kitti.positions(poses), yaw_deg(poses)
    db, q = kitti.split_indices(seq, len(poses))
    top1 = None
    if matches_dir:
        mp = os.path.join(matches_dir, f"eval_{seq}_matches.npz")
        if os.path.exists(mp):
            m = np.load(mp)
            top1 = dict(zip(m["query"].tolist(), m["top1"].tolist()))
    rows = []
    for qi in q:
        d = np.linalg.norm(xyz[db] - xyz[qi], axis=1)
        pos = np.flatnonzero(d < thr)
        if len(pos) == 0:
            continue
        nearest = pos[np.argmin(d[pos])]
        diffs = ang_diff(yaw[qi], yaw[db[pos]])
        row = {"seq": seq, "query": int(qi), "nearest_db": int(db[nearest]),
               "dist_nearest_m": float(d[nearest]),
               "diff_nearest_deg": float(ang_diff(yaw[qi], yaw[db[nearest]])),
               "diff_min_deg": float(diffs.min())}
        if top1 is not None and int(qi) in top1:
            row["top1_correct"] = bool(np.linalg.norm(xyz[top1[int(qi)]] - xyz[qi]) < thr)
        rows.append(row)
    return pd.DataFrame(rows), len(q)


def group_of(x):
    for name, lo, hi in GROUPS:
        if lo <= x < hi:
            return name
    return GROUPS[-1][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seqs", nargs="+", default=["00", "02", "05", "06", "08"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold", type=float, default=kitti.MATCH_THRESHOLD_M)
    ap.add_argument("--matches_dir", default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows, summary = [], []
    for seq in args.seqs:
        df, n_q = analyse(args.kitti_root, seq, args.threshold, args.matches_dir)
        df["group"] = df["diff_nearest_deg"].map(group_of)
        all_rows.append(df)
        s = {"seq": seq, "queries": n_q, "queries_with_positive": len(df),
             "median_diff_nearest_deg": round(float(df["diff_nearest_deg"].median()), 1),
             "median_diff_min_deg": round(float(df["diff_min_deg"].median()), 1)}
        for name, _, _ in GROUPS:
            g = df[df["group"] == name]
            s[f"% {name}"] = round(100 * len(g) / max(len(df), 1), 1)
            if "top1_correct" in df:
                s[f"R@1 {name}"] = round(100 * g["top1_correct"].mean(), 1) if len(g) else None
        summary.append(s)

    per_query = pd.concat(all_rows, ignore_index=True)
    per_query.to_csv(os.path.join(args.out_dir, "heading_per_query.csv"), index=False)
    table = pd.DataFrame(summary)
    table.to_csv(os.path.join(args.out_dir, "heading_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "heading_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    n = len(args.seqs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), sharey=False, squeeze=False)
    bins = np.arange(0, 181, 10)
    for ax, df, seq in zip(axes[0], all_rows, args.seqs):
        ax.hist(df["diff_nearest_deg"], bins=bins, color="tab:blue", alpha=0.8,
                label="to nearest map frame")
        ax.hist(df["diff_min_deg"], bins=bins, histtype="step", color="tab:red", lw=1.5,
                label="smallest within 5 m")
        ax.set_title(f"KITTI {seq} ({len(df)} queries)")
        ax.set_xlabel("heading difference (deg)")
        ax.set_xticks([0, 45, 90, 135, 180])
    axes[0][0].set_ylabel("queries")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "heading_hist.png"), dpi=120)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.to_string(index=False))
    print("->", args.out_dir)


if __name__ == "__main__":
    main()
