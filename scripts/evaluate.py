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
    ext = "*.png" if method == "bev" else "*.bin"
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
            script = "eval_bev_kitti.py" if a.method == "bev" else "eval_ndt_kitti.py"
            arg = "--bev_dir" if a.method == "bev" else "--ndt_dir"
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


def yaw_eval(a, ckpt, out, seqs, levels, key, yaw_files, title, xlabel):
    """Shared by compass and rotation: for every sequence, the database rotated by one .npy of
    angles and the queries by one .npy per level; results in <out>/<key>_<level>/, a summary
    with a column <key> and a plot of Recall@1 against the level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ydir = os.path.join(out, "yaw")
    os.makedirs(ydir, exist_ok=True)
    lev_dir = {v: os.path.join(out, f"{key.split('_')[0]}_{v:g}") for v in levels}
    for d in lev_dir.values():
        os.makedirs(d, exist_ok=True)
    done, missing = [], []
    for seq in seqs:
        if not has_data(a.method, a.data_dir, seq):
            missing.append(seq)
            continue
        ext = "*.png" if a.method == "bev" else "*.bin"
        n = len(glob.glob(os.path.join(a.data_dir, seq, ext)))
        outs = [os.path.join(d, f"eval_{seq}.json") for d in lev_dir.values()]
        if a.force or not all(os.path.exists(o) and evals.ckpt_name(json.load(open(o))["ckpt"])
                              == evals.ckpt_name(ckpt) for o in outs):
            db_file, q_files = yaw_files(seq, n, ydir)
            script = "eval_bev_kitti.py" if a.method == "bev" else "eval_ndt_kitti.py"
            arg = "--bev_dir" if a.method == "bev" else "--ndt_dir"
            print(f"  {seq}: evaluating {len(q_files)} levels of {key}", flush=True)
            run([PY, os.path.join(ROOT, "scripts", script), "--ckpt", ckpt, arg, a.data_dir,
                 "--kitti_root", a.kitti_root, "--seq", seq,
                 "--yaw_db", db_file, "--yaw_q", *q_files, "--outs", *outs])
        done.append(seq)
    if missing:
        print("  no data for sequences", missing, "(skipped; the evaluation stays incomplete)")
    if not done:
        return False
    tables = []
    for v, d in lev_dir.items():
        run([PY, os.path.join(ROOT, "scripts", "heading_analysis.py"), "--kitti_root", a.kitti_root,
             "--seqs", *done, "--out_dir", d, "--matches_dir", d])
        evals.write_seqs_summary(d, done)
        t = pd.read_csv(os.path.join(d, "summary.csv"), dtype={"seq": str})
        t.insert(0, key, v)
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(os.path.join(out, "summary.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    for seq, g in table.groupby("seq"):
        ax.plot(g[key], g["R@1"], "o-", label=f"KITTI {seq}")
    ax.set_xlabel(xlabel); ax.set_ylabel("Recall@1 (%)")
    ax.set_title(f"{title}, {a.method}: {os.path.basename(os.path.normpath(a.run_dir))}", fontsize=9)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out, f"{title}.png"), dpi=120)
    return not missing


def compass(a, params, ckpt, out):
    """Map and queries rotated by their heading (one orientation); queries also get a compass
    error drawn from N(0, sigma^2)."""
    import numpy as np
    from lpr import kitti

    def yaw_files(seq, n, ydir):
        yaw = kitti.velo_yaw_deg(a.kitti_root, seq)[:n]
        db_file = os.path.join(ydir, f"map_{seq}.npy")
        np.save(db_file, yaw)
        q_files = []
        for s in params["sigma_deg"]:
            rng = np.random.default_rng([params["seed"], int(seq), int(round(s * 100))])
            f = os.path.join(ydir, f"query_{seq}_sigma_{s:g}.npy")
            np.save(f, yaw + rng.normal(0.0, s, n))
            q_files.append(f)
        return db_file, q_files

    return yaw_eval(a, ckpt, out, params["seqs"], params["sigma_deg"], "sigma_deg", yaw_files,
                    "compass", "compass error sigma (deg)")


def rotation(a, params, ckpt, out):
    """Map as recorded; every query rotated about z by its own angle from U[-alpha, alpha]."""
    import numpy as np

    def yaw_files(seq, n, ydir):
        db_file = os.path.join(ydir, f"map_{seq}.npy")
        np.save(db_file, np.zeros(n))
        q_files = []
        for al in params["alpha_deg"]:
            rng = np.random.default_rng([params["seed"], int(seq), int(round(al * 100))])
            f = os.path.join(ydir, f"query_{seq}_alpha_{al:g}.npy")
            np.save(f, rng.uniform(-al, al, n))
            q_files.append(f)
        return db_file, q_files

    return yaw_eval(a, ckpt, out, params["seqs"], params["alpha_deg"], "alpha_deg", yaw_files,
                    "rotation", "query rotation amplitude alpha (deg)")


def sparsity(a, params, ckpt, out):
    """Recall with 64 beams (the run's variant) and with fewer beams (its sparse variants,
    folders next to --data_dir), map and queries sparse ("both") or queries only ("query")."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lpr import experiments as ex
    data = "bev" if a.method == "bev" else "ndt"
    base = os.path.basename(os.path.normpath(a.data_dir))
    try:
        base_params = json.load(open(os.path.join(a.data_dir, "config.json")))["params"]
    except (OSError, ValueError, KeyError):
        raise SystemExit(f"no config.json with the parameters of the variant in {a.data_dir}")
    if base_params.get("beams", 64) != 64:
        raise SystemExit("sparsity starts from a 64-beam variant")
    script = "eval_bev_kitti.py" if a.method == "bev" else "eval_ndt_kitti.py"
    arg = "--bev_dir" if a.method == "bev" else "--ndt_dir"

    def run_one(folder, db_dir, query_dir, seqs):
        os.makedirs(folder, exist_ok=True)
        done, missing = [], []
        for s in seqs:
            if not (has_data(a.method, db_dir, s) and has_data(a.method, query_dir, s)):
                missing.append(s)
                continue
            res = os.path.join(folder, f"eval_{s}.json")
            same = os.path.exists(res) and \
                evals.ckpt_name(json.load(open(res)).get("ckpt", "")) == evals.ckpt_name(ckpt)
            if not same or a.force:
                print(f"  {os.path.basename(folder)} {s}: evaluating", flush=True)
                cmd = [PY, os.path.join(ROOT, "scripts", script), "--ckpt", ckpt, arg, db_dir,
                       "--kitti_root", a.kitti_root, "--seq", s, "--out", res]
                if query_dir != db_dir:
                    cmd += ["--query_dir", query_dir]
                run(cmd)
            done.append(s)
        if done:
            run([PY, os.path.join(ROOT, "scripts", "heading_analysis.py"), "--kitti_root",
                 a.kitti_root, "--seqs", *done, "--out_dir", folder, "--matches_dir", folder])
            evals.write_seqs_summary(folder, done)
        return done, missing

    tables, all_missing = [], []
    done, missing = run_one(os.path.join(out, "beams64"), a.data_dir, a.data_dir, params["seqs"])
    all_missing += [("64", s) for s in missing]
    if done:
        t = pd.read_csv(os.path.join(out, "beams64", "summary.csv"), dtype={"seq": str})
        for mode in params["modes"]:   # 64 beams: both modes are the same evaluation
            tables.append(t.assign(beams=64, mode=mode))
    for b in params["beams"]:
        name, _ = ex.sparse_variant(data, base, base_params, b)
        sdir = os.path.join(os.path.dirname(os.path.normpath(a.data_dir)), name)
        if not os.path.isdir(sdir):
            print(f"  {b} beams: variant {name} not found at {sdir} (skipped)")
        for mode in params["modes"]:
            folder = os.path.join(out, f"beams{b}_{mode}")
            db_dir = sdir if mode == "both" else a.data_dir
            done, missing = run_one(folder, db_dir, sdir, params["seqs"])
            all_missing += [(f"{b} {mode}", s) for s in missing]
            if done:
                t = pd.read_csv(os.path.join(folder, "summary.csv"), dtype={"seq": str})
                tables.append(t.assign(beams=b, mode=mode, variant=name))
    if all_missing:
        print("  no data for", all_missing, "(skipped; the evaluation stays incomplete)")
    if not tables:
        return False
    table = pd.concat(tables, ignore_index=True)
    cols = ["beams", "mode"] + [c for c in table.columns if c not in ("beams", "mode")]
    table = table[cols].sort_values(["mode", "seq", "beams"], ascending=[True, True, False])
    table.to_csv(os.path.join(out, "summary.csv"), index=False)
    fig, axes = plt.subplots(1, len(params["modes"]), figsize=(6 * len(params["modes"]), 4),
                             squeeze=False)
    for ax, mode in zip(axes[0], params["modes"]):
        for seq, g in table[table["mode"] == mode].groupby("seq"):
            ax.plot(g["beams"], g["R@1"], "o-", label=f"KITTI {seq}")
        ax.set_xticks([64] + list(params["beams"]))
        ax.invert_xaxis()
        ax.set_xlabel("laser beams")
        ax.set_ylabel("Recall@1 (%)")
        ax.set_title({"both": "map and queries sparse", "query": "map 64 beams, queries sparse"}
                     .get(mode, mode))
        ax.legend()
    fig.suptitle(f"sparsity, {a.method}: {os.path.basename(os.path.normpath(a.run_dir))}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sparsity.png"), dpi=120)
    return not all_missing


FUNCS = {"seqs_heading": seqs_heading, "tnet": tnet, "compass": compass, "sparsity": sparsity,
         "rotation": rotation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ndt", "bev", "myndt"])
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
