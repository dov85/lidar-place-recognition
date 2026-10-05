"""One table of all runs: parameters, NDT variant and results.

Reads <drive>/runs/*/config.json, metrics.csv and eval_*.json, and writes
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

KEY_PARAMS = ["lr", "epochs", "negatives", "positives", "margin1", "margin2", "pos_thr",
              "neg_thr", "hard_neg_after_epoch", "amp", "seed"]
KEY_NDT = ["crop", "half_size", "ground", "radius", "over", "min_pts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.drive, "runs", "*"))):
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            continue
        cfg = json.load(open(cfg_path))
        row = {"run": os.path.basename(run_dir), "status": cfg.get("status"),
               "ndt_variant": cfg.get("ndt_variant"), "gpu": cfg.get("gpu"),
               "started": cfg.get("started"), "commit": cfg.get("code_commit")}
        row.update({k: cfg.get("params", {}).get(k) for k in KEY_PARAMS})
        ndt_path = os.path.join(run_dir, "ndt_config.json")
        if os.path.exists(ndt_path):
            ndt = json.load(open(ndt_path)).get("params", {})
            row.update({f"ndt_{k}": ndt.get(k) for k in KEY_NDT})
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

    df = pd.DataFrame(rows)
    out = os.path.join(args.drive, "runs_summary.csv")
    df.to_csv(out, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df)
    print("->", out)


if __name__ == "__main__":
    main()
