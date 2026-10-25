"""Bookkeeping for preprocessing variants and training runs on Google Drive.

Three methods share the layout: "ndt" (NDT-Transformer, NDT cells), "bev"
(BEVPlace++, BEV density images) and "myndt" (own NDT network trained with the BEVPlace++
recipe; uses the NDT variants of "ndt").
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
    beams=64,             # 32 / 16: rings of the HDL-64E dropped before anything else (sparsity
                          # axis, src/lpr/beams.py)
)

BEV_DEFAULTS = dict(       # official BEVPlace++ values (datasets/gen_bev_images.py)
    half_size=40.0,
    z_abs=40.0,
    voxel=0.4,
    res=0.4,
    max_count=10,
    ground="none",
    beams=64,             # 32 / 16: fewer laser rings (src/lpr/beams.py)
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
    hard_neg_after_epoch=0,  # N > 0: random negatives in epochs 1..N, hard mining from epoch N+1
                           # (as NDT-Transformer, N=5); 0 = hard mining from the start (official)
    beam_aug=[],          # e.g. [64, 32, 16]: every training sample from one of these numbers of laser
                          # beams, drawn at random (the 32 / 16-beam variants of the run's variant,
                          # experiments.sparse_variant); [] = only the run's own variant
    arch="rein",           # "rein": the official network (8 rotations of the image through the
                           # CNN, max over rotations, NetVLAD); "plain": the same CNN and NetVLAD
                           # with one pass and no rotations (no built-in rotation invariance)
)

METHODS = {
    # method: (preprocessing defaults, training defaults, archive prefix, variant key in run hash)
    "ndt": ("NDT_DEFAULTS", "TRAIN_DEFAULTS", "cells", "ndt_variant"),
    "bev": ("BEV_DEFAULTS", "BEV_TRAIN_DEFAULTS", "imgs", "bev_variant"),
    # own NDT network (src/lpr/models) trained with the BEVPlace++ recipe; uses the NDT variants
    "myndt": ("NDT_DEFAULTS", "MYNDT_TRAIN_DEFAULTS", "cells", "ndt_variant"),
}

# Own NDT network (src/lpr/models/ndt_net.py) with the training recipe of BEVPlace++
# (src/lpr/bevplace.py, the same code as BEV runs): only the network and its input differ.
MYNDT_TRAIN_DEFAULTS = dict(
    seq="00",
    train_frames=3000,     # recipe of BEVPlace++ (as BEV_TRAIN_DEFAULTS) ...
    pos_thr=5.0,
    neg_thr=7.0,
    epochs=40,
    lr=1e-4,
    batch=4,               # queries per batch, each with 1 positive and 10 negatives
    negatives=10,
    margin=0.3,            # triplet loss, hardest negative; hard mining from a descriptor cache
                           # of all training frames at the start of every epoch
    rot_aug=True,          # random 0-360 deg rotation of every training cloud (cells R m, R S R^T)
    seed=1024,
    hard_neg_after_epoch=0,  # N > 0: random negatives in epochs 1..N, hard mining from epoch N+1
                           # (as NDT-Transformer, N=5); 0 = hard mining from the start (BEVPlace++)
    rot_points=0,          # N > 0: training clouds from the N point-rotated copies of the variant
                           # (as TRAIN_DEFAULTS rot_points); with rot_aug the copy is rotated too
    beam_aug=[],           # e.g. [64, 32, 16]: every training sample from one of these numbers of laser
                          # beams, drawn at random (the 32 / 16-beam variants of the run's variant,
                          # experiments.sparse_variant); [] = only the run's own variant

    amp=True,              # mixed precision (GPU memory; 8 rotations x 2000 cells per cloud)
    # ... network:
    stn=False,             # T-Net (STN3d) of NDT-Transformer before the encoder
    rotations=8,           # REIN-like: cells rotated by 0, 45, ..., 315 deg, the same encoder on
                           # every copy, maximum over the copies per cell; 1 = no rotations
    head="vlad",           # "vlad": NetVLAD of BEVPlace++ (64 x feat_dim, k-means init);
                           # "loupe": NetVLADLoupe of NDT-Transformer (256, with a final L2)
    feat_dim=128,          # channels per cell before the head (BEVPlace++: 128; official NDT: 1024)
    cov_transform=False,   # with stn: the T-Net also transforms the covariances
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
    cov_transform=False,  # official code: False (T-Net applied to the cell means only);
                          # True = also T^T Sigma T on the covariances, as in the paper
    rot_aug=False,        # True: every training cloud rotated about z by its own random angle
                          # (0-360 deg; means R m, covariances R S R^T)
    rot_points=0,         # N > 0: every training cloud is taken from one of N copies of its scan
                          # whose raw points were rotated by a random angle before the cells were
                          # built (cells_<seq>_rot<N>.tar of the variant, built on Azure)
    beam_aug=[],          # e.g. [64, 32, 16]: every training sample from one of these numbers of laser
                          # beams, drawn at random (the 32 / 16-beam variants of the run's variant,
                          # experiments.sparse_variant); [] = only the run's own variant
)

# Parameters added after runs already existed: left out of the run hash while they have
# these values, so the existing runs keep their names (see run_hash).
HASH_OMIT_IF = {"ndt": {"cov_transform": False, "rot_aug": False, "rot_points": 0,
                        "beam_aug": []},
                "bev": {"arch": "rein", "hard_neg_after_epoch": 0, "beam_aug": []},
                "myndt": {"hard_neg_after_epoch": 0, "beam_aug": []}}


# Preprocessing parameters added after variants already existed: left out of the variant
# hash while they have these values, so the existing variants keep their names.
VARIANT_HASH_OMIT_IF = {"beams": 64}


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
    cfg["params_hash"] = ndt_hash(params)
    return cfg, True


def method_defaults(method):
    pre, train, _, _ = METHODS[method]
    return globals()[pre], globals()[train]


def archive_name(method, seq):
    return f"{METHODS[method][2]}_{seq}.tar"


def rot_dir_name(seq, copies):
    """Folder (inside the variant) of the point-rotated copies of one sequence."""
    return f"{seq}_rot{int(copies)}"


def rot_archive_name(seq, copies):
    return f"cells_{rot_dir_name(seq, copies)}.tar"


def resolve_ndt_variant(drive, params, tag=""):
    return resolve_variant(drive, "ndt", params, tag)


def resolve_variant(drive, method, params, tag=""):
    """Existing variant with exactly these parameters, or a new name for them."""
    defaults = method_defaults(method)[0]
    p = full_params(defaults, params)
    h = ndt_hash(p)
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


def sparse_variant(method, base_variant, base_params, beams):
    """Name and parameters of the variant with `beams` laser rings derived from a 64-beam
    variant (sparsity axis): the same parameters with beams changed, named
    <date of the base variant>_beams<N>_<hash>, so Colab and the Azure VM agree on it."""
    defaults = method_defaults(method)[0]
    p = full_params(defaults, dict(base_params, beams=beams))
    return f"{base_variant[:10]}_beams{beams}_{ndt_hash(p)}", p


def beam_aug_dirs(data_dir, method, beams):
    """Folders of the training samples for beam_aug: the variant itself for 64 beams, its
    sparse variants (next to it, as fetch_sparse.py puts them) for fewer."""
    base = os.path.basename(os.path.normpath(data_dir))
    params = json.load(open(os.path.join(data_dir, "config.json")))["params"]
    return [data_dir if int(b) == 64 else
            os.path.join(os.path.dirname(os.path.normpath(data_dir)),
                         sparse_variant(method, base, params, int(b))[0]) for b in beams]


def ndt_hash(ndt_params_full):
    """Hash of a preprocessing variant (NDT or BEV)."""
    p = {k: v for k, v in ndt_params_full.items()
         if not (k in VARIANT_HASH_OMIT_IF and v == VARIANT_HASH_OMIT_IF[k])}
    return _hash(p)
