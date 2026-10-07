"""Bookkeeping for preprocessing variants and training runs on Google Drive.

Two methods share the layout: "ndt" (NDT-Transformer, NDT cells) and "bev"
(BEVPlace++, BEV density images).
  <drive>/<method>/<variant>/config.json   every preprocessing parameter, and per
                                            sequence: scans, stats, date ("seqs")
                    <archive>_<seq>.tar    one sequence (cells_XX.tar for ndt,
                                            imgs_XX.tar for bev)
                    stats_<seq>.csv        per-scan statistics
  <drive>/runs/<method>/<run>/config.json  training parameters, variant, status
                     <method>_config.json  copy of the variant's config
                     metrics.csv           per-epoch loss / recall / time
                     epoch_XX.pth, latest.pth
  <drive>/runs/<method>/<run>/eval/<type>/<checkpoint>/  one evaluation (src/lpr/evals.py)
  <drive>/runs_summary.csv                 one row per run (column "method")

Names are "<date>_<tag>_<hash>": the hash comes from the parameters, so the same
parameters always find the same folder (a disconnected run resumes), and different
parameters always get a new one. A variant does not depend on the sequence: the
same parameters applied to 00, 02, ... live in one folder, one tar per sequence.
"""
import datetime
import glob
import hashlib
import json
import os
import subprocess

NDT_DEFAULTS = dict(
    crop="square",        # square (as BEVPlace++) or circle
    half_size=40.0,       # m; also the coordinate normalisation scale
    ground="patchwork",   # patchwork | ransac | none
    k=2000,
    over=0.10,
    radius=0.8,
    min_pts=5,
    knn=8,
    cov_eps=1e-3,
)

BEV_DEFAULTS = dict(       # official BEVPlace++ values (datasets/gen_bev_images.py)
    half_size=40.0,
    z_abs=40.0,
    voxel=0.4,
    res=0.4,
    max_count=10,
    ground="none",
)

BEV_TRAIN_DEFAULTS = dict(  # official BEVPlace++ values (main.py, kitti_dataset.py)
    seq="00",
    train_frames=3000,     # the first 3000 frames of the sequence
    pos_thr=5.0,
    neg_thr=7.0,
    epochs=40,
    lr=1e-4,
    batch=4,               # triplets per batch (query, positive, 10 negatives)
    negatives=10,
    margin=0.3,
    rot_aug=True,          # random 0-360 deg rotation of every training image
    seed=1024,
)

METHODS = {
    # method: (preprocessing defaults, training defaults, archive prefix, variant key in run hash)
    "ndt": ("NDT_DEFAULTS", "TRAIN_DEFAULTS", "cells", "ndt_variant"),
    "bev": ("BEV_DEFAULTS", "BEV_TRAIN_DEFAULTS", "imgs", "bev_variant"),
}

TRAIN_DEFAULTS = dict(
    seq="00",
    pos_thr=5.0,          # tuple thresholds (make_kitti_tuples.py)
    neg_thr=7.0,
    epochs=20,
    lr=1e-5,
    milestones=[9, 15],
    batch_queries=2,
    positives=2,
    negatives=18,
    margin1=0.5,
    margin2=0.2,
    hard_neg_after_epoch=5,
    amp=True,
    seed=0,
    cov_transform=False,  # official code: False (T-Net applied to the cell means only);
                          # True = also T^T Sigma T on the covariances, as in the paper
)

# Parameters added after runs already existed: left out of the run hash while they have
# these values, so the existing runs keep their names (see run_hash).
HASH_OMIT_IF = {"ndt": {"cov_transform": False}, "bev": {}}


def _hash(params):
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:6]


def _read(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def git_commit(repo_dir):
    try:
        return subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def full_params(defaults, params):
    unknown = set(params) - set(defaults)
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}; allowed: {sorted(defaults)}")
    out = dict(defaults)
    for k, v in params.items():
        d = defaults[k]
        if isinstance(d, bool):
            out[k] = bool(v)
        elif isinstance(d, float):
            out[k] = float(v)
        elif isinstance(d, int):
            out[k] = int(v)
        elif isinstance(d, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _new_name(tag, h, is_default):
    """<date>_baseline_<hash> for the default parameters; otherwise <date>_<tag>_<hash>
    with the tag given in the notebook, or <date>_<hash> without one. "baseline" is
    reserved for the defaults, so a changed experiment never gets that name."""
    date = datetime.date.today().isoformat()
    if is_default:
        tag = "baseline"
    elif tag == "baseline":
        print('note: "baseline" is only used for the default parameters; naming this one by date and hash')
        tag = ""
    return f"{date}_{tag}_{h}" if tag else f"{date}_{h}"


def upgrade_ndt_config(cfg):
    """Configs written before variants became sequence-independent had "seq" in the
    parameters (and in the hash). Move it to cfg["seqs"] and recompute the hash."""
    params = cfg.get("params", {})
    if "seq" not in params:
        return cfg, False
    seq = params.pop("seq")
    per_seq = {k: cfg.pop(k) for k in ("n_scans", "stats_mean", "padded_scans", "started",
                                       "finished") if k in cfg}
    cfg.setdefault("seqs", {})[seq] = per_seq
    cfg["params_hash"] = _hash(params)
    return cfg, True


def method_defaults(method):
    pre, train, _, _ = METHODS[method]
    return globals()[pre], globals()[train]


def archive_name(method, seq):
    return f"{METHODS[method][2]}_{seq}.tar"


def resolve_ndt_variant(drive, params, tag=""):
    return resolve_variant(drive, "ndt", params, tag)


def resolve_variant(drive, method, params, tag=""):
    """Existing variant with exactly these parameters, or a new name for them."""
    defaults = method_defaults(method)[0]
    p = full_params(defaults, params)
    h = _hash(p)
    for cfg_path in sorted(glob.glob(os.path.join(drive, method, "*", "config.json"))):
        cfg = _read(cfg_path)
        if cfg:
            cfg, changed = upgrade_ndt_config(cfg)
            if changed:
                write_json(cfg_path, cfg)
        if cfg and cfg.get("params_hash") == h:
            name = os.path.basename(os.path.dirname(cfg_path))
            return name, p, True
    return _new_name(tag, h, p == full_params(defaults, {})), p, False


def resolve_run(drive, train_params, ndt_variant, tag="", method="ndt"):
    """Run folder (under <drive>/runs/<method>/) for these parameters and variant.

    state "resume": an unfinished run exists and is continued;
    state "finished": a finished run exists; training is skipped and only evaluation runs
                      (to repeat it on purpose, change the seed);
    state "new": nothing with these parameters yet.
    """
    defaults = method_defaults(method)[1]
    p = full_params(defaults, train_params)
    h = run_hash(p, ndt_variant, method)
    finished_twin = None
    for cfg_path in sorted(glob.glob(os.path.join(drive, "runs", method, "*", "config.json"))):
        cfg = _read(cfg_path)
        if cfg and cfg.get("params_hash") == h:
            name = os.path.basename(os.path.dirname(cfg_path))
            if cfg.get("status") != "finished":
                return name, p, "resume"
            finished_twin = name
    if finished_twin:
        return finished_twin, p, "finished"
    return _new_name(tag, h, p == full_params(defaults, {})), p, "new"


def run_hash(train_params_full, variant, method="ndt"):
    omit = HASH_OMIT_IF[method]
    p = {k: v for k, v in train_params_full.items() if not (k in omit and v == omit[k])}
    return _hash({"train": p, METHODS[method][3]: variant})


def ndt_hash(ndt_params_full):
    return _hash(ndt_params_full)
