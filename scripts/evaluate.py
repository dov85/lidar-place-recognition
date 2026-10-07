"""Run one evaluation type (src/lpr/evals.py) on one checkpoint of a run.

Results: <run_dir>/eval/<eval_type>/<checkpoint>/ (config.json, summary.csv, ...).
A complete evaluation with the same parameters and checkpoint is not repeated.

Examples:
  python scripts/evaluate.py --method ndt --run_dir <drive>/runs/ndt/<run> --eval_type seqs_heading \
      --data_dir /content/ndt/<variant> --kitti_root /content/kitti
  python scripts/evaluate.py --method bev --run_dir <drive>/runs/bev/official_<variant> \
      --ckpt third_party/BEVPlace2/runs/Aug08_10-17-29/model_best.pth.tar --eval_type seqs_heading \
      --data_dir /content/bev/<variant> --kitti_root /content/kitti
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from lpr import evals  # noqa: E402
from lpr.experiments import git_commit  # noqa: E402

PY = sys.executable


def run(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if res.returncode != 0:
        print(res.stdout[-1500:], res.stderr[-3000:])
        raise SystemExit(f"failed: {' '.join(cmd[:3])} ...")
    return res.stdout


def has_data(method, data_dir, seq):
    ext = "*.bin" if method == "ndt" else "*.png"
    return len(glob.glob(os.path.join(data_dir, seq, ext))) > 0


def seqs_heading(a, params, ckpt, out):
    done, missing = [], []
    for s in params["seqs"]:
        if not has_data(a.method, a.data_dir, s):
            missing.append(s)
            continue
        res_json = os.path.join(out, f"eval_{s}.json")
        same = os.path.exists(res_json) and \
            evals.ckpt_name(json.load(open(res_json)).get("ckpt", "")) == evals.ckpt_name(ckpt)
        if not same or a.force:
            script = "eval_ndt_kitti.py" if a.method == "ndt" else "eval_bev_kitti.py"
            arg = "--ndt_dir" if a.method == "ndt" else "--bev_dir"
            print(f"  {s}: evaluating", flush=True)
            run([PY, os.path.join(ROOT, "scripts", script), "--ckpt", ckpt, arg, a.data_dir,
                 "--kitti_root", a.kitti_root, "--seq", s, "--out", res_json])
        done.append(s)
    if missing:
        print("  no data for sequences", missing, "(skipped; the evaluation stays incomplete)")
    if not done:
        return False
    run([PY, os.path.join(ROOT, "scripts", "heading_analysis.py"), "--kitti_root", a.kitti_root,
         "--seqs", *done, "--out_dir", out, "--matches_dir", out])
    evals.write_seqs_summary(out, done)
    return not missing


def tnet(a, params, ckpt, out):
    if a.method != "ndt":
        raise SystemExit("tnet is for NDT-Transformer runs only")
    if not has_data("ndt", a.data_dir, params["seq"]):
        print(f"  no NDT cells for sequence {params['seq']}")
        return False
    run([PY, os.path.join(ROOT, "scripts", "tnet_check.py"), "--ckpt", ckpt, "--ndt_dir", a.data_dir,
         "--seq", params["seq"], "--n_scans", str(params["n_scans"]),
         "--angles", *map(str, params["angles"]), "--out_dir", out])
    return True


def compass(a, params, ckpt, out):
    import numpy as np
    from lpr import kitti
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ydir = os.path.join(out, "yaw")
    os.makedirs(ydir, exist_ok=True)
    sig_dir = {s: os.path.join(out, f"sigma_{s:g}") for s in params["sigma_deg"]}
    for d in sig_dir.values():
        os.makedirs(d, exist_ok=True)
    done, missing = [], []
    for seq in params["seqs"]:
        if not has_data(a.method, a.data_dir, seq):
            missing.append(seq)
            continue
        ext = "*.bin" if a.method == "ndt" else "*.png"
        n = len(glob.glob(os.path.join(a.data_dir, seq, ext)))
        outs = [os.path.join(d, f"eval_{seq}.json") for d in sig_dir.values()]
        if a.force or not all(os.path.exists(o) and evals.ckpt_name(json.load(open(o))["ckpt"])
                              == evals.ckpt_name(ckpt) for o in outs):
            yaw = kitti.velo_yaw_deg(a.kitti_root, seq)[:n]
            np.save(os.path.join(ydir, f"map_{seq}.npy"), yaw)
            qfiles = []
            for s in params["sigma_deg"]:
                rng = np.random.default_rng([params["seed"], int(seq), int(round(s * 100))])
                f = os.path.join(ydir, f"query_{seq}_sigma_{s:g}.npy")
                np.save(f, yaw + rng.normal(0.0, s, n))
                qfiles.append(f)
            script = "eval_ndt_kitti.py" if a.method == "ndt" else "eval_bev_kitti.py"
            arg = "--ndt_dir" if a.method == "ndt" else "--bev_dir"
            print(f"  {seq}: evaluating {len(qfiles)} compass errors", flush=True)
            run([PY, os.path.join(ROOT, "scripts", script), "--ckpt", ckpt, arg, a.data_dir,
                 "--kitti_root", a.kitti_root, "--seq", seq,
                 "--yaw_db", os.path.join(ydir, f"map_{seq}.npy"), "--yaw_q", *qfiles, "--outs", *outs])
        done.append(seq)
    if missing:
        print("  no data for sequences", missing, "(skipped; the evaluation stays incomplete)")
    if not done:
        return False
    import pandas as pd
    tables = []
    for s, d in sig_dir.items():
        run([PY, os.path.join(ROOT, "scripts", "heading_analysis.py"), "--kitti_root", a.kitti_root,
             "--seqs", *done, "--out_dir", d, "--matches_dir", d])
        evals.write_seqs_summary(d, done)
        t = pd.read_csv(os.path.join(d, "summary.csv"), dtype={"seq": str})
        t.insert(0, "sigma_deg", s)
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(os.path.join(out, "summary.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    for seq, g in table.groupby("seq"):
        ax.plot(g["sigma_deg"], g["R@1"], "o-", label=f"KITTI {seq}")
    ax.set_xlabel("compass error sigma (deg)"); ax.set_ylabel("Recall@1 (%)")
    ax.set_title(f"compass, {a.method}: {os.path.basename(os.path.normpath(a.run_dir))}", fontsize=9)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out, "compass.png"), dpi=120)
    return not missing


FUNCS = {"seqs_heading": seqs_heading, "tnet": tnet, "compass": compass}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ndt", "bev"])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="last", help="last | epoch_18.pth | a path")
    ap.add_argument("--eval_type", required=True, choices=sorted(evals.EVAL_TYPES))
    ap.add_argument("--data_dir", required=True, help="NDT cells or BEV images of the variant")
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--params", default="{}", help="JSON overrides of the type's parameters")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.method not in evals.EVAL_TYPES[a.eval_type]["methods"]:
        print(f"{a.eval_type}: not defined for {a.method}, skipped")
        return
    params = evals.full_eval_params(a.eval_type, json.loads(a.params))
    ckpt = evals.resolve_ckpt(a.run_dir, a.ckpt)
    out = evals.eval_dir(a.run_dir, a.eval_type, ckpt)
    print(f"{a.eval_type} | {os.path.basename(os.path.normpath(a.run_dir))} | {evals.ckpt_name(ckpt)}")
    if evals.is_done(out, params) and not a.force:
        print("  already done ->", out)
    else:
        evals.write_config(out, a.eval_type, params, ckpt, a.run_dir, False, git_commit(ROOT))
        complete = FUNCS[a.eval_type](a, params, ckpt, out)
        evals.write_config(out, a.eval_type, params, ckpt, a.run_dir, complete, git_commit(ROOT))
        print("  ->", out, "" if complete else "(incomplete)")
    s = os.path.join(out, "summary.csv")
    if os.path.exists(s):
        with pd.option_context("display.max_columns", None, "display.width", 250):
            print(pd.read_csv(s, dtype={"seq": str}).to_string(index=False))


if __name__ == "__main__":
    main()
