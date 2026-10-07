"""One table of all runs (NDT-Transformer and BEVPlace++): parameters, variant, results.

Reads <drive>/runs/<method>/*/config.json, metrics.csv and eval/<type>/<checkpoint>/, and writes
  <drive>/runs_summary.csv    one row per run (parameters, training, the latest seqs_heading
                              evaluation as columns eval_XX_R@1 ...)
  <drive>/evals_summary.csv   every evaluation: method, run, eval_type, checkpoint, then the
                              rows of its summary.csv (one per sequence, angle, ...)

Example:
  python scripts/summarize_runs.py --drive /content/drive/MyDrive/lpr
"""
import argparse
import glob
import json
import os

import pandas as pd

KEY_PARAMS = ["lr", "epochs", "negatives", "positives", "margin", "margin1", "margin2", "batch",
              "pos_thr", "neg_thr", "hard_neg_after_epoch", "rot_aug", "cov_transform", "amp", "seed"]
KEY_PRE = {"ndt": ["crop", "half_size", "ground", "radius", "over", "min_pts"],
           "bev": ["half_size", "voxel", "res", "max_count", "ground"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True)
    args = ap.parse_args()

    rows, eval_rows = [], []
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
        for out in sorted(glob.glob(os.path.join(run_dir, "eval", "*", "*", "summary.csv"))):
            ckpt_dir = os.path.dirname(out)
            etype, ckpt = os.path.basename(os.path.dirname(ckpt_dir)), os.path.basename(ckpt_dir)
            try:
                complete = json.load(open(os.path.join(ckpt_dir, "config.json"))).get("complete")
            except (OSError, ValueError):
                complete = None
            t = pd.read_csv(out, dtype={"seq": str})
            t.insert(0, "complete", complete)
            t.insert(0, "checkpoint", ckpt)
            t.insert(0, "eval_type", etype)
            t.insert(0, "run", row["run"])
            t.insert(0, "method", method)
            eval_rows.append(t)
            if etype == "seqs_heading" and "seq" in t:   # latest checkpoint wins (sorted names)
                row["eval_checkpoint"] = ckpt
                for _, r in t.iterrows():
                    row[f"eval_{r['seq']}_R@1"] = r["R@1"]
                    row[f"eval_{r['seq']}_R@1%"] = r["R@1%"]
        rows.append(row)

    if eval_rows:
        ev = pd.concat(eval_rows, ignore_index=True)
        ev.to_csv(os.path.join(args.drive, "evals_summary.csv"), index=False)
        print("->", os.path.join(args.drive, "evals_summary.csv"), f"({len(ev)} rows)")
    df = pd.DataFrame(rows).dropna(axis=1, how="all")
    out = os.path.join(args.drive, "runs_summary.csv")
    df.to_csv(out, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df)
    print("->", out)


if __name__ == "__main__":
    main()
