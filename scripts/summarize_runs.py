"""One table of all runs (NDT-Transformer and BEVPlace++): parameters, variant, results.

Reads <drive>/runs/<method>/*/config.json, metrics.csv and eval_*.json, and writes
<drive>/runs_summary.csv (open it in Google Sheets or with pandas).

Example:
  python scripts/summarize_runs.py --drive /content/drive/MyDrive/lpr
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

KEY_PARAMS = ["lr", "epochs", "negatives", "positives", "margin", "margin1", "margin2", "batch",
              "pos_thr", "neg_thr", "hard_neg_after_epoch", "rot_aug", "cov_transform", "amp", "seed"]
KEY_PRE = {"ndt": ["crop", "half_size", "ground", "radius", "over", "min_pts"],
           "bev": ["half_size", "voxel", "res", "max_count", "ground"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    rows = []
    run_dirs = [(m, d) for m in ("ndt", "bev")
                for d in sorted(glob.glob(os.path.join(args.drive, "runs", m, "*")))]
    for method, run_dir in run_dirs:
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        cfg = json.load(open(cfg_path))
        row = {"method": method, "run": os.path.basename(run_dir), "status": cfg.get("status"),
               "variant": cfg.get(f"{method}_variant"), "gpu": cfg.get("gpu"),
               "started": cfg.get("started"), "commit": cfg.get("code_commit")}
        row.update({k: cfg.get("params", {}).get(k) for k in KEY_PARAMS})
        pre_path = os.path.join(run_dir, f"{method}_config.json")
        if os.path.exists(pre_path):
            pre = json.load(open(pre_path)).get("params", {})
            row.update({f"pre_{k}": pre.get(k) for k in KEY_PRE[method]})
        m_path = os.path.join(run_dir, "metrics.csv")
        if os.path.exists(m_path):
            m = pd.read_csv(m_path).drop_duplicates("epoch", keep="last")
            last = m.iloc[-1]
            row.update({"epochs_done": int(last["epoch"]),
                        "train_hours": round(m["epoch_minutes"].sum() / 60, 2),
                        "last_R@1": last.get("R@1"), "last_R@1%": last.get("R@1%"),
                        "best_R@1": m["R@1"].max() if "R@1" in m else None})
        for e in sorted(glob.glob(os.path.join(run_dir, "eval_*.json"))):
            tag = os.path.splitext(os.path.basename(e))[0]
            if not re.fullmatch(r"eval_\d\d", tag):   # eval_00.json, eval_08.json, ...
                continue
            r = json.load(open(e))
            row[f"{tag}_R@1"] = r.get("R@1")
            row[f"{tag}_R@1%"] = r.get("R@1%")
        rows.append(row)

    df = pd.DataFrame(rows).dropna(axis=1, how="all")
    out = os.path.join(args.drive, "runs_summary.csv")
    df.to_csv(out, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df)
    print("->", out)


if __name__ == "__main__":
    main()
