"""One-off: move the first NDT variant and the first training run into the new layout.

Before                                   After
  <drive>/ndt_kitti00_patchwork.tar  ->  <drive>/ndt/<variant>/cells_00.tar (+ config.json)
  <drive>/00_stats.csv               ->  <drive>/ndt/<variant>/stats_00.csv
  <drive>/runs/ndt_kitti00/          ->  <drive>/runs/ndt/<run>/ (+ config.json, ndt_config.json)

The first variant used the default parameters (patchwork, 80 m square, G defaults)
and the first run the default training parameters, so the hashes match what
experiments.resolve_* compute for the defaults.

Run without --apply first to see the plan.
  python scripts/migrate_stage1.py --drive /content/drive/MyDrive/lpr
  python scripts/migrate_stage1.py --drive /content/drive/MyDrive/lpr --apply
"""
import argparse
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr.experiments import (NDT_DEFAULTS, TRAIN_DEFAULTS, full_params, ndt_hash,  # noqa: E402
                             run_hash, write_json)

STAGE1_COMMIT = "63d57d2"
STAGE1_DATE = "2026-10-04"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    ap.add_argument("--old_tar", default="ndt_kitti00_patchwork.tar")
    ap.add_argument("--old_stats", default="00_stats.csv")
    ap.add_argument("--old_run", default="runs/ndt_kitti00")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    d = a.drive

    ndt_params = full_params(NDT_DEFAULTS, {})
    variant = f"{STAGE1_DATE}_baseline_{ndt_hash(ndt_params)}"
    vdir = os.path.join(d, "ndt", variant)

    old_run = os.path.join(d, a.old_run)
    old_args = json.load(open(os.path.join(old_run, "args.json")))
    train_given = {k: old_args[k] for k in TRAIN_DEFAULTS if k in old_args}
    train_params = full_params(TRAIN_DEFAULTS, train_given)
    run = f"{STAGE1_DATE}_baseline_{run_hash(train_params, variant)}"
    rdir = os.path.join(d, "runs", "ndt", run)

    old_tar, old_stats = os.path.join(d, a.old_tar), os.path.join(d, a.old_stats)
    plan = [(old_tar, os.path.join(vdir, "cells_00.tar")),
            (old_stats, os.path.join(vdir, "stats_00.csv")),
            (old_run, rdir)]
    print("NDT variant:", variant)
    print("run:        ", run)
    for src, dst in plan:
        print(("move " if os.path.exists(src) else "MISSING ") + src + "\n  -> " + dst)
    if not a.apply:
        print("\n(dry run; add --apply to move the files)")
        return

    os.makedirs(vdir, exist_ok=True)
    os.makedirs(os.path.dirname(rdir), exist_ok=True)
    for src, dst in plan:
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)

    stats_path = os.path.join(vdir, "stats_00.csv")
    stats_mean = {}
    if os.path.exists(stats_path):
        st = pd.read_csv(stats_path)
        stats_mean = st[["n_raw", "n_points", "voxel", "cells_before_removal", "removed",
                         "padded", "seconds"]].mean().round(4).to_dict()
    ndt_cfg = {"variant": variant, "params": ndt_params, "params_hash": ndt_hash(ndt_params),
               "seqs": {"00": {"code_commit": STAGE1_COMMIT, "started": STAGE1_DATE,
                               "stats_mean": stats_mean}},
               "migrated": True}
    write_json(os.path.join(vdir, "config.json"), ndt_cfg)

    metrics = pd.read_csv(os.path.join(rdir, "metrics.csv")).drop_duplicates("epoch", keep="last")
    done = int(metrics["epoch"].max())
    run_cfg = {"run": run, "ndt_variant": variant, "params": train_params,
               "params_hash": run_hash(train_params, variant),
               "tuples": {"seq": "00", "split": 3000, "pos_thr": 5.0, "neg_thr": 7.0},
               "code_commit": STAGE1_COMMIT, "gpu": "Tesla T4", "started": STAGE1_DATE,
               "status": "finished" if done >= train_params["epochs"] else "running",
               "migrated": True}
    write_json(os.path.join(rdir, "config.json"), run_cfg)
    write_json(os.path.join(rdir, "ndt_config.json"), ndt_cfg)
    for f in os.listdir(rdir):  # old evaluation file name -> eval_00.json
        if f.startswith("eval_00_") and f.endswith(".json"):
            shutil.copy(os.path.join(rdir, f), os.path.join(rdir, "eval_00.json"))
    print("done")


if __name__ == "__main__":
    main()
