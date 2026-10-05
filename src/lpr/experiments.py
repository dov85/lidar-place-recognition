"""Bookkeeping for NDT variants and training runs on Google Drive.

Layout
  <drive>/ndt/<variant>/config.json        every preprocessing + G parameter
                        cells_<seq>.tar     the NDT cells
                        stats_<seq>.csv     per-scan statistics
                        preview_<seq>.png   cells drawn on one scan
  <drive>/runs/<run>/config.json            training parameters, NDT variant, status
                     ndt_config.json        copy of the variant's config
                     metrics.csv            per-epoch loss / recall / time
                     epoch_XX.pth, latest.pth, eval_<seq>.json
  <drive>/runs_summary.csv                  one row per run

Names are "<date>_<tag>_<hash>": the hash comes from the parameters, so the same
parameters always find the same folder (a disconnected run resumes), and different
parameters always get a new one.
"""
import datetime
import glob
import hashlib
import json
import os
import subprocess

NDT_DEFAULTS = dict(
    seq="00",
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
)


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


def _new_name(tag, h):
    date = datetime.date.today().isoformat()
    return f"{date}_{tag}_{h}" if tag else f"{date}_{h}"


def resolve_ndt_variant(drive, params, tag=""):
    """Existing variant with exactly these parameters, or a new name for them."""
    p = full_params(NDT_DEFAULTS, params)
    h = _hash(p)
    for cfg_path in sorted(glob.glob(os.path.join(drive, "ndt", "*", "config.json"))):
        cfg = _read(cfg_path)
        if cfg and cfg.get("params_hash") == h:
            name = os.path.basename(os.path.dirname(cfg_path))
            return name, p, True
    return _new_name(tag, h), p, False


def resolve_run(drive, train_params, ndt_variant, tag=""):
    """Run folder for these parameters and NDT variant.

    state "resume": an unfinished run exists and is continued;
    state "finished": a finished run exists; training is skipped and only evaluation runs
                      (to repeat it on purpose, change the seed);
    state "new": nothing with these parameters yet.
    """
    p = full_params(TRAIN_DEFAULTS, train_params)
    key = {"train": p, "ndt_variant": ndt_variant}
    h = _hash(key)
    finished_twin = None
    for cfg_path in sorted(glob.glob(os.path.join(drive, "runs", "*", "config.json"))):
        cfg = _read(cfg_path)
        if cfg and cfg.get("params_hash") == h:
            name = os.path.basename(os.path.dirname(cfg_path))
            if cfg.get("status") != "finished":
                return name, p, "resume"
            finished_twin = name
    if finished_twin:
        return finished_twin, p, "finished"
    return _new_name(tag, h), p, "new"


def run_hash(train_params_full, ndt_variant):
    return _hash({"train": train_params_full, "ndt_variant": ndt_variant})


def ndt_hash(ndt_params_full):
    return _hash(ndt_params_full)
