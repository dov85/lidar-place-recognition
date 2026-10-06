"""Train NDT-Transformer on KITTI with the official network, loss and tuple sampling.

What comes from the official repository (third_party/NDT-Transformer):
  models.NDTNetVlad, loss.quadruplet_loss (lazy), loading_pointclouds
  (file loader, get_query_tuple) and the hard-negative mining scheme of train.py.
What is new here: the training loop around them (checkpoint/resume that survives
Colab disconnects, optional mixed precision and split forward for 16 GB GPUs,
timing logs) and evaluation on KITTI instead of Oxford.

Hyper-parameters follow the paper: 20 epochs, batch of 2 queries, 2 positives,
18 negatives, 1 other negative, margins 0.5 / 0.2, lr 1e-5 decayed x0.1 at
epochs 9 and 15.

Example:
  python scripts/train_ndt_kitti.py --ndt_dir /content/ndt --kitti_root /content/kitti \
      --train_pickle /content/ndt/train_00.pickle --log_dir /content/drive/MyDrive/lpr/runs/ndt_00
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "NDT-Transformer"))

import config as cfg  # noqa: E402  (official config module)
import loading_pointclouds as lp  # noqa: E402
import loss.loss as official_loss  # noqa: E402
import models.NDTNetVlad as official_model  # noqa: E402
from sklearn.neighbors import KDTree  # noqa: E402

from lpr import kitti  # noqa: E402
from lpr.evaluation import recall  # noqa: E402
from lpr.experiments import TRAIN_DEFAULTS, full_params, git_commit, run_hash, write_json  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CELLS = 2000
DESC_DIM = 256


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndt_dir", required=True)
    ap.add_argument("--kitti_root", required=True, help="for poses (evaluation)")
    ap.add_argument("--train_pickle", required=True)
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--milestones", type=int, nargs="+", default=[9, 15])
    ap.add_argument("--batch_queries", type=int, default=2)
    ap.add_argument("--positives", type=int, default=2)
    ap.add_argument("--negatives", type=int, default=18)
    ap.add_argument("--margin1", type=float, default=0.5)
    ap.add_argument("--margin2", type=float, default=0.2)
    ap.add_argument("--hard_neg_after_epoch", type=int, default=5, help="official: epoch > 5")
    ap.add_argument("--amp", action="store_true", help="mixed precision (saves GPU memory)")
    ap.add_argument("--split_forward", action="store_true",
                    help="one query tuple per forward pass; gradients are accumulated")
    ap.add_argument("--ckpt_every", type=int, default=300, help="iterations")
    ap.add_argument("--eval_every", type=int, default=1, help="epochs; 0 = never")
    ap.add_argument("--max_iters", type=int, default=None, help="per epoch, for smoke tests")
    ap.add_argument("--eval_split", type=int, default=None, help="override split (tests)")
    ap.add_argument("--eval_gap", type=int, default=kitti.QUERY_GAP, help="with --eval_split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cov_transform", action="store_true",
                    help="T-Net also transforms the covariances (T^T Sigma T), as in the paper; "
                         "the official code leaves this off")
    ap.add_argument("--ndt_variant", default="", help="name of the NDT variant (recorded)")
    return ap.parse_args()


# ----------------------------------------------------------------------------- data
def load_tuple_batch(queries, keys, args, latent):
    """Official get_query_tuple, with the official hard-negative logic of train.py."""
    tuples = []
    for key in keys:
        entry = queries[key]
        if len(entry["positives"]) < args.positives:
            return None
        hard = []
        if latent is not None:
            random.shuffle(entry["negatives"])
            cand = entry["negatives"][:4000]
            q_vec = descriptors([entry["query"]], model_cache["model"])[0]
            _, ind = KDTree(latent[cand]).query(q_vec[None], k=min(10, len(cand)))
            hard = np.array(cand)[ind[0]].tolist()
        t = lp.get_query_tuple(entry, args.positives, args.negatives, queries,
                               hard_neg=hard, other_neg=True)
        if t[3].shape[0] != NUM_CELLS:
            return None
        tuples.append(t)
    return tuples


model_cache = {}


@torch.no_grad()
def descriptors(files, model, batch=32):
    was_training = model.training
    model.eval()
    out = []
    for s in range(0, len(files), batch):
        x = torch.from_numpy(lp.load_pc_files(files[s:s + batch])).float().unsqueeze(1).to(DEVICE)
        out.append(model(x).float().cpu().numpy())
    model.train(was_training)
    return np.concatenate(out).reshape(len(files), -1)


def forward_tuples(model, tuples, args):
    """Returns the loss of a list of tuples (query, positives, negatives, other_neg)."""
    q = np.stack([t[0] for t in tuples])[:, None]
    p = np.stack([t[1] for t in tuples])
    n = np.stack([t[2] for t in tuples])
    o = np.stack([t[3] for t in tuples])[:, None]
    feed = torch.from_numpy(np.concatenate([q, p, n, o], axis=1)).float()
    b = feed.shape[0]
    feed = feed.view(-1, 1, NUM_CELLS, 12).to(DEVICE)
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=args.amp):
        out = model(feed)
    out = out.float().view(b, -1, DESC_DIM)
    o1, o2, o3, o4 = torch.split(out, [1, args.positives, args.negatives, 1], dim=1)
    return official_loss.quadruplet_loss(o1, o2, o3, o4, args.margin1, args.margin2,
                                         use_min=False, lazy=True, ignore_zero_loss=False)


def train_step(model, tuples, args, scaler):
    """Forward + backward; with split_forward the gradient is accumulated per tuple."""
    if not args.split_forward:
        l = forward_tuples(model, tuples, args)
        scaler.scale(l).backward()
        return l.item()
    total = 0.0
    for t in tuples:
        l = forward_tuples(model, [t], args) / len(tuples)
        scaler.scale(l).backward()
        total += l.item()
    return total


# ----------------------------------------------------------------------------- eval
def evaluate(model, args):
    n = len(glob.glob(os.path.join(args.ndt_dir, args.seq, "*.bin")))
    files = [os.path.join(args.seq, f"{i:06d}.bin") for i in range(n)]
    t0 = time.time()
    desc = descriptors(files, model)
    xyz = kitti.positions(kitti.load_poses(args.kitti_root, args.seq))[:n]
    if args.eval_split:
        db = np.arange(args.eval_split)
        qi = np.arange(args.eval_split + args.eval_gap, n)
    else:
        db, qi = kitti.split_indices(args.seq, n)
    r = recall(desc[db], desc[qi], xyz[db], xyz[qi], threshold=kitti.MATCH_THRESHOLD_M)
    r.pop("top1_indices")
    r["descriptor_seconds_per_scan"] = (time.time() - t0) / n
    return r, desc


# ----------------------------------------------------------------------------- ckpt
def save_ckpt(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def run_config(args):
    """Full training parameters (same keys as experiments.TRAIN_DEFAULTS) and their hash."""
    tuples_json = os.path.splitext(args.train_pickle)[0] + ".json"
    tuples = json.load(open(tuples_json)) if os.path.exists(tuples_json) else {}
    given = {k: getattr(args, k) for k in TRAIN_DEFAULTS if hasattr(args, k)}
    given.update({k: tuples[k] for k in ("pos_thr", "neg_thr") if k in tuples})
    params = full_params(TRAIN_DEFAULTS, given)
    return params, run_hash(params, args.ndt_variant), tuples


def main():
    args = parse()
    os.makedirs(args.log_dir, exist_ok=True)
    params, phash, tuples = run_config(args)
    cfg_path = os.path.join(args.log_dir, "config.json")
    old = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
    if old is not None and old.get("params_hash") != phash:
        raise SystemExit(
            f"{args.log_dir} already holds a run with different parameters "
            f"(hash {old.get('params_hash')} != {phash}). Use a new log_dir / run name.")
    run_cfg = old or {
        "run": os.path.basename(os.path.normpath(args.log_dir)),
        "ndt_variant": args.ndt_variant,
        "params": params,
        "params_hash": phash,
        "tuples": tuples,
        "paths": {"ndt_dir": args.ndt_dir, "train_pickle": args.train_pickle},
        "code_commit": git_commit(ROOT),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sessions": 0,
    }
    run_cfg["status"] = "running"
    run_cfg["sessions"] = run_cfg.get("sessions", 0) + 1
    write_json(cfg_path, run_cfg)
    ndt_cfg = os.path.join(args.ndt_dir, "config.json")
    if os.path.exists(ndt_cfg):
        write_json(os.path.join(args.log_dir, "ndt_config.json"), json.load(open(ndt_cfg)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg.DATASET_FOLDER = args.ndt_dir
    cfg.NUM_POINTS = NUM_CELLS

    queries = lp.get_queries_dict(args.train_pickle)
    model = official_model.NDTNetVlad(num_points=NUM_CELLS, output_dim=DESC_DIM,
                                      emb_dims=1024, layer_number=3).to(DEVICE)
    model.Transformer.convariance_transform = args.cov_transform  # official switch, off by default
    model_cache["model"] = model
    opt = torch.optim.Adam(model.parameters(), args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=0.1)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=args.amp)

    latest = os.path.join(args.log_dir, "latest.pth")
    epoch, start_iter, perm = 0, 0, None
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        epoch, start_iter, perm = ck["epoch"], ck["iter"], ck["perm"]
        print(f"resumed from {latest}: epoch {epoch}, iteration {start_iter}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {DEVICE}, parameters {n_params / 1e6:.2f} M, training queries {len(queries)}")

    metrics_path = os.path.join(args.log_dir, "metrics.csv")
    latent = None
    keys_all = np.array(sorted(queries.keys()))
    while epoch < args.epochs:
        if perm is None:
            perm = np.random.permutation(keys_all)
        n_iters = len(perm) // args.batch_queries
        if args.max_iters:
            n_iters = min(n_iters, args.max_iters)
        model.train()
        t_epoch, t_log, losses, skipped, it_log = time.time(), time.time(), [], 0, start_iter
        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for i in range(start_iter, n_iters):
            # official schedule: refresh the cached descriptors for hard negatives
            if epoch > args.hard_neg_after_epoch and i % (1400 // args.batch_queries) == 29:
                latent = descriptors([queries[k]["query"] for k in keys_all], model)
            keys = perm[i * args.batch_queries:(i + 1) * args.batch_queries]
            tuples = load_tuple_batch(queries, keys, args, latent)
            if tuples is None:
                skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            try:
                total = train_step(model, tuples, args, scaler)
            except torch.cuda.OutOfMemoryError:
                if args.split_forward:
                    raise
                # 2 x 22 clouds do not fit: fall back to one tuple per forward pass
                print("CUDA out of memory with the full batch -> switching to --split_forward",
                      flush=True)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                args.split_forward = True
                total = train_step(model, tuples, args, scaler)
            scaler.step(opt)
            scaler.update()
            losses.append(total)
            if (i + 1) % 50 == 0 or i == n_iters - 1:
                dt = (time.time() - t_log) / max(i + 1 - it_log, 1)
                mem = torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else 0
                eta = dt * (n_iters - i - 1) / 60
                print(f"epoch {epoch} it {i + 1}/{n_iters} loss {np.mean(losses[-50:]):.4f} "
                      f"{dt:.2f}s/it (~{eta:.0f} min left in epoch) peak GPU mem {mem:.1f} GB "
                      f"lr {opt.param_groups[0]['lr']:.1e}", flush=True)
                t_log, it_log = time.time(), i + 1
            if (i + 1) % args.ckpt_every == 0:
                save_ckpt(latest, model=model.state_dict(), optimizer=opt.state_dict(),
                          scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                          epoch=epoch, iter=i + 1, perm=perm)
        epoch_min = (time.time() - t_epoch) / 60
        sched.step()
        epoch += 1
        start_iter, perm = 0, None
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"),
               "epoch_minutes": round(epoch_min, 2), "skipped_batches": skipped,
               "split_forward": args.split_forward, "amp": args.amp}
        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
            r, _ = evaluate(model, args)
            row.update({k: r[k] for k in ("R@1", "R@5", "R@1%", "n_queries_with_positive")})
        print("EPOCH SUMMARY", json.dumps(row), flush=True)
        new = not os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        state = dict(model=model.state_dict(), optimizer=opt.state_dict(),
                     scheduler=sched.state_dict(), scaler=scaler.state_dict(),
                     epoch=epoch, iter=0, perm=None)
        save_ckpt(latest, **state)
        save_ckpt(os.path.join(args.log_dir, f"epoch_{epoch:02d}.pth"), model=model.state_dict(),
                  epoch=epoch)
    run_cfg["status"] = "finished"
    run_cfg["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(cfg_path, run_cfg)
    print("training finished")


if __name__ == "__main__":
    main()
