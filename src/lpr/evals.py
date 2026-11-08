"""Evaluation types and where their results live.

Every evaluation of a run is stored in
  <run_dir>/eval/<eval_type>/<checkpoint>/
      config.json    eval type, its parameters, checkpoint, date, code version, "complete"
      summary.csv    one table per evaluation (collected into <drive>/evals_summary.csv)
      ...            files of that evaluation type
<checkpoint> is the checkpoint file name without extension (epoch_20, model_best, ...).
An evaluation that is complete with the same parameters and checkpoint is not run again;
any change of either gives a new or a refreshed folder.

New types: add an entry to EVAL_TYPES and a function to scripts/evaluate.py.
"""
import datetime
import glob
import json
import os

ALL_SEQS = ["00", "02", "05", "06", "08"]

EVAL_TYPES = {
    # Recall on every sequence (BEVPlace++ protocol) + query headings vs map headings,
    # with Recall@1 per heading group (same / crossing / opposite).
    "seqs_heading": dict(methods=["ndt", "bev", "myndt"],
                         params=dict(seqs=ALL_SEQS)),
    # NDT-Transformer only: what the T-Net learned. Scans rotated about the vertical axis;
    # distance of T from the identity, whether T follows the rotation, descriptor drift.
    "tnet": dict(methods=["ndt"],
                 params=dict(seq="08", n_scans=200, angles=[0, 15, 30, 45, 60, 90, 120, 150, 180])),
    # Compass, no retraining: every scan rotated about z by its heading from the ground-truth
    # pose, so map and queries share one orientation; each query also gets a compass error
    # drawn from N(0, sigma^2), for each sigma (degrees). Recall per sequence and sigma, and
    # per heading group (relative to the true headings).
    # Fewer laser beams (sparsity axis), no retraining: the run's checkpoint on the variants of
    # its 64-beam variant with 32 / 16 beams (experiments.sparse_variant; built on Azure).
    # mode "both": map and queries from the sparse sensor; "query": map from the 64-beam
    # variant, queries from the sparse one. 64 beams (the run's own variant) is included.
    "sparsity": dict(methods=["ndt", "bev", "myndt"],
                     params=dict(seqs=ALL_SEQS, beams=[32, 16], modes=["both", "query"])),
    "compass": dict(methods=["ndt", "bev", "myndt"],
                    params=dict(seqs=ALL_SEQS, sigma_deg=[0, 2, 5, 10, 20, 45], seed=0)),
    # Rotation curve, no retraining: the map stays as recorded; every query is rotated about z
    # by its own angle drawn from U[-alpha, alpha] (alpha = 180: any heading). Recall per
    # sequence and alpha, and per heading group (relative to the true headings).
    "rotation": dict(methods=["ndt", "bev", "myndt"],
                     params=dict(seqs=ALL_SEQS, alpha_deg=[0, 5, 15, 45, 90, 180], seed=0)),
}


def ckpt_name(ckpt_path):
    name = os.path.basename(ckpt_path)
    for ext in (".pth.tar", ".pth", ".tar"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def resolve_ckpt(run_dir, ckpt="last"):
    """'last' -> highest epoch_XX.pth of the run; a file name -> inside the run; a path -> itself."""
    if ckpt == "last":
        eps = sorted(glob.glob(os.path.join(run_dir, "epoch_*.pth")))
        if not eps:
            raise SystemExit(f"no epoch_XX.pth in {run_dir}: nothing to evaluate")
        return eps[-1]
    path = ckpt if os.path.isabs(ckpt) else os.path.join(run_dir, ckpt)
    if not os.path.exists(path):
        raise SystemExit(f"checkpoint not found: {path}")
    return path


def eval_dir(run_dir, eval_type, ckpt_path):
    return os.path.join(run_dir, "eval", eval_type, ckpt_name(ckpt_path))


def full_eval_params(eval_type, overrides=None):
    if eval_type not in EVAL_TYPES:
        raise SystemExit(f"unknown eval type {eval_type}; known: {sorted(EVAL_TYPES)}")
    p = dict(EVAL_TYPES[eval_type]["params"])
    p.update(overrides or {})
    return p


def is_done(out_dir, params):
    try:
        cfg = json.load(open(os.path.join(out_dir, "config.json")))
    except (OSError, ValueError):
        return False
    return cfg.get("complete") and cfg.get("params") == params


def write_config(out_dir, eval_type, params, ckpt_path, run_dir, complete, commit=""):
    from lpr.experiments import write_json
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "config.json"), {
        "eval_type": eval_type, "params": params, "checkpoint": ckpt_name(ckpt_path),
        "ckpt_path": ckpt_path, "run": os.path.basename(os.path.normpath(run_dir)),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "code_commit": commit, "complete": complete})


def write_seqs_summary(out_dir, seqs):
    """summary.csv of a seqs_heading evaluation: recall per sequence + heading groups."""
    import pandas as pd
    head_path = os.path.join(out_dir, "heading_summary.csv")
    head = pd.read_csv(head_path, dtype={"seq": str}) if os.path.exists(head_path) else None
    if head is not None:
        head["seq"] = head["seq"].str.zfill(2)
    rows = []
    for s in seqs:
        r = json.load(open(os.path.join(out_dir, f"eval_{s}.json")))
        row = {"seq": s, **{k: round(r[k], 2) for k in ("R@1", "R@5", "R@1%")},
               "queries": r["n_queries_with_positive"]}
        if head is not None:
            h = head[head["seq"] == s]
            if len(h):
                row.update({c: h.iloc[0][c] for c in head.columns
                            if c.startswith("% ") or c.startswith("R@1 ") or c.startswith("median")})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
