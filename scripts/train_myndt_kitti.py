"""Train the own NDT network (src/lpr/models/ndt_net.py) with the BEVPlace++ recipe.

Everything except the network and its input is the training code of the BEV runs
(src/lpr/bevplace.py, as train_bev_kitti.py): frames 0-2999 of the sequence, positives < 5 m,
negatives > 7 m, 10 negatives per query, batch 4, triplet loss with margin 0.3 on the
hardest negative, hard mining from a descriptor cache of all training frames rebuilt at the
start of every epoch (optionally only from epoch hard_neg_after_epoch + 1, random negatives
before, as NDT-Transformer does), Adam lr 1e-4, 40 epochs, random rotation of every training sample,
NetVLAD initialised by k-means (head "vlad").

Network options (MYNDT_TRAIN_DEFAULTS in src/lpr/experiments.py): stn (T-Net), rotations
(REIN-like rotation equivariance, 8 = 45 deg steps), head (vlad | loupe), feat_dim,
cov_transform. Run bookkeeping as the other runs: config.json (every parameter, hash, code
commit, GPU), ndt_config.json (the variant), metrics.csv, epoch_XX.pth, latest.pth (resume,
also inside an epoch).

Example:
  python scripts/train_myndt_kitti.py --ndt_dir /content/ndt/<variant> --kitti_root /content/kitti \
      --ndt_variant <variant> --log_dir /content/drive/MyDrive/lpr/runs/myndt/<run>
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from lpr import bevplace as bp  # noqa: E402
from lpr import kitti, myndt  # noqa: E402
from lpr.evaluation import recall  # noqa: E402
from lpr.experiments import (MYNDT_TRAIN_DEFAULTS, beam_aug_dirs, full_params, git_commit,  # noqa: E402
                             run_hash, write_json)
from lpr.models import ndt_net  # noqa: E402
from lpr.models.heads import HEADS  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the transformer's inference fast path (descriptor cache, evaluation) builds full attention
# matrices; the regular path uses memory-efficient attention, as in training
torch.backends.mha.set_fastpath_enabled(False)


def parse():
    d = MYNDT_TRAIN_DEFAULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndt_dir", required=True, help="variant folder: <seq>/<idx>.bin")
    ap.add_argument("--kitti_root", required=True, help="for poses")
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--ndt_variant", default="")
    for k in ("seq", "train_frames", "pos_thr", "neg_thr", "epochs", "lr", "batch", "negatives",
              "margin", "seed", "hard_neg_after_epoch", "rot_points", "rotations", "feat_dim"):
        ap.add_argument(f"--{k}", type=type(d[k]), default=d[k])
    ap.add_argument("--head", default=d["head"], choices=list(HEADS))
    ap.add_argument("--beam_aug", type=int, nargs="*", default=[], choices=[64, 32, 16],
                    help="e.g. 64 32 16: every training sample from one of these numbers of beams")
    for k in ("rot_aug", "amp", "stn", "cov_transform"):   # booleans: --x / --no_x
        ap.add_argument(f"--{k}", dest=k, action="store_true")
        ap.add_argument(f"--no_{k}", dest=k, action="store_false")
        ap.set_defaults(**{k: d[k]})
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=1, help="epochs; 0 = never")
    ap.add_argument("--no_eval_at_start", dest="eval_at_start", action="store_false",
                    help="skip the recall before training (epoch 0 of metrics.csv)")
    ap.add_argument("--eval_batch", type=int, default=16,
                    help="clouds per forward pass when computing descriptors (memory only)")
    ap.add_argument("--ckpt_every", type=int, default=150, help="iterations")
    ap.add_argument("--max_iters", type=int, default=0, help="per epoch, testing only")
    ap.add_argument("--n_cluster_desc", type=int, default=10000, help="testing only")
    ap.add_argument("--split_forward", action="store_true",
                    help="one tuple per forward pass from the start (otherwise only after an OOM)")
    a = ap.parse_args()
    if a.cov_transform and not a.stn:
        ap.error("--cov_transform needs --stn (the T-Net transforms the covariances)")
    return a


def save_ckpt(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def evaluate(model, args, paths, xyz):
    t0 = time.time()
    desc = bp.infer(model, myndt.InferDataset(paths), DEVICE, batch=args.eval_batch,
                    workers=args.workers, amp=args.amp)
    db, q = kitti.split_indices(args.seq, len(paths))
    r = recall(desc[db], desc[q], xyz[db], xyz[q], threshold=kitti.MATCH_THRESHOLD_M)
    r.pop("top1_indices")
    r["descriptor_seconds_per_scan"] = (time.time() - t0) / len(paths)
    return r


def main():
    args = parse()
    os.makedirs(args.log_dir, exist_ok=True)
    params = full_params(MYNDT_TRAIN_DEFAULTS,
                         {k: getattr(args, k) for k in MYNDT_TRAIN_DEFAULTS})
    phash = run_hash(params, args.ndt_variant, "myndt")
    cfg_path = os.path.join(args.log_dir, "config.json")
    old = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
    if old is not None and old.get("params_hash") != phash:
        raise SystemExit(f"{args.log_dir} already holds a run with different parameters "
                         f"(hash {old.get('params_hash')} != {phash}). Use a new log_dir / run name.")
    run_cfg = old or {
        "run": os.path.basename(os.path.normpath(args.log_dir)),
        "method": "myndt",
        "ndt_variant": args.ndt_variant,
        "params": params,
        "params_hash": phash,
        "paths": {"ndt_dir": args.ndt_dir},
        "code_commit": git_commit(ROOT),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sessions": 0,
    }
    run_cfg["status"] = "running"
    run_cfg["sessions"] = run_cfg.get("sessions", 0) + 1
    write_json(cfg_path, run_cfg)
    var_cfg = os.path.join(args.ndt_dir, "config.json")
    if os.path.exists(var_cfg):
        write_json(os.path.join(args.log_dir, "ndt_config.json"), json.load(open(var_cfg)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    paths = myndt.cell_paths(args.ndt_dir, args.seq)
    poses = kitti.load_poses(args.kitti_root, args.seq)[:len(paths)]
    xyz = kitti.positions(poses)
    train_set = myndt.TrainingDataset(args.ndt_dir, args.seq, poses.reshape(len(poses), 12),
                                      args.train_frames, args.pos_thr, args.neg_thr,
                                      args.negatives, args.rot_aug, args.rot_points)
    if args.beam_aug:   # the same scans with other numbers of beams (fetch_sparse.py)
        if getattr(args, "rot_points", 0):
            raise SystemExit("beam_aug and rot_points cannot be combined")
        train_set.beam_dirs = beam_aug_dirs(args.ndt_dir, "ndt", args.beam_aug)
        for d in train_set.beam_dirs:
            have = len(os.listdir(os.path.join(d, args.seq))) if os.path.isdir(os.path.join(d, args.seq)) else 0
            if have < len(train_set):
                raise SystemExit(f"beam_aug: {d}/{args.seq} has {have} files, {len(train_set)} needed "
                                 "(copy the sparse variants with scripts/fetch_sparse.py)")
        print("beam_aug: training samples from", [os.path.basename(d) for d in train_set.beam_dirs])
    model = ndt_net.build(params).to(DEVICE)
    print(f"network: T-Net {'on' if args.stn else 'off'}, {args.rotations} rotation(s) per "
          f"cloud, head {args.head}, {args.feat_dim} channels per cell, descriptor "
          f"{model.global_feat_dim}")

    def new_optimizer():
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    opt = new_optimizer()
    scaler = torch.amp.GradScaler("cuda") if args.amp and DEVICE.type == "cuda" else None
    latest = os.path.join(args.log_dir, "latest.pth")
    epoch, start_iter, perm = 0, 0, None
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        epoch, start_iter, perm = ck["epoch"], ck["iter"], ck["perm"]
        np.random.seed(args.seed + 1000 * epoch + start_iter)
        print(f"resumed from {latest}: epoch {epoch}, iteration {start_iter}")
    elif hasattr(model.pooling, "init_params"):
        print("NetVLAD initialisation: k-means on cell features of the training clouds")
        t0 = time.time()
        centroids, descs = bp.get_clusters(model, train_set, DEVICE, batch=16,
                                           workers=args.workers,
                                           n_descriptors=args.n_cluster_desc, seed=args.seed)
        np.savez(os.path.join(args.log_dir, "netvlad_init.npz"), centroids=centroids)
        model.pooling.init_params(centroids, descs)
        model = model.to(DEVICE)
        opt = new_optimizer()
        print(f"  done in {time.time() - t0:.0f} s")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {DEVICE}, parameters {n_params / 1e6:.2f} M, training frames {len(train_set)}, "
          f"clouds {len(paths)}, mixed precision {'on' if scaler else 'off'}")

    def state(ep, it, pm):
        return dict(model=model.state_dict(), optimizer=opt.state_dict(),
                    scaler=scaler.state_dict() if scaler else None, epoch=ep, iter=it, perm=pm)

    metrics_path = os.path.join(args.log_dir, "metrics.csv")
    if args.eval_at_start and epoch == 0 and start_iter == 0 and not os.path.exists(metrics_path):
        r = evaluate(model, args, paths, xyz)   # after the NetVLAD initialisation, before training
        row = {"epoch": 0, **{k: r[k] for k in ("R@1", "R@5", "R@1%", "n_queries_with_positive")}}
        print("BEFORE TRAINING", json.dumps(row), flush=True)
        bp.append_metrics(metrics_path, row)
    split = [args.split_forward]
    while epoch < args.epochs:
        t_epoch = time.time()
        if perm is None:
            perm = np.random.permutation(len(train_set))
        batches = [perm[i:i + args.batch].tolist() for i in range(0, len(perm), args.batch)]
        if args.max_iters:
            batches = batches[:args.max_iters]
        mining = epoch + 1 > args.hard_neg_after_epoch
        t0 = time.time()
        if mining:
            print(f"epoch {epoch + 1}: descriptor cache for hard mining", flush=True)
            bp.build_cache(model, train_set, DEVICE, batch=args.batch, workers=args.workers,
                           amp=args.amp)
        else:   # warm-up: random negatives, the nearest positive (the recipe's start-up mode)
            print(f"epoch {epoch + 1}: random negatives (hard mining from epoch "
                  f"{args.hard_neg_after_epoch + 1})", flush=True)
            train_set.mining = False
        cache_min = (time.time() - t0) / 60
        t_log = [time.time(), start_iter]

        def log(it, losses):
            if it % 50 == 0 or it == len(batches):
                dt = (time.time() - t_log[0]) / max(it - t_log[1], 1)
                mem = torch.cuda.max_memory_allocated() / 2**30 if DEVICE.type == "cuda" else 0
                print(f"epoch {epoch + 1} it {it}/{len(batches)} loss {np.mean(losses[-50:]):.4f} "
                      f"{dt:.2f}s/it (~{dt * (len(batches) - it) / 60:.0f} min left in epoch) "
                      f"peak GPU mem {mem:.1f} GB", flush=True)
                t_log[:] = [time.time(), it]

        def on_iter(it, losses):
            if it % args.ckpt_every == 0 and it < len(batches):
                save_ckpt(latest, **state(epoch, it, perm))

        losses = bp.train_batches(model, train_set, batches, opt, DEVICE, args.batch, args.margin,
                                  args.workers, log, on_iter, start=start_iter, split=split,
                                  scaler=scaler)
        epoch += 1
        start_iter, perm = 0, None
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"),
               "epoch_minutes": round((time.time() - t_epoch) / 60, 2),
               "cache_minutes": round(cache_min, 2), "split_forward": split[0],
               "hard_mining": mining}
        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
            r = evaluate(model, args, paths, xyz)
            row.update({k: r[k] for k in ("R@1", "R@5", "R@1%", "n_queries_with_positive")})
        print("EPOCH SUMMARY", json.dumps(row), flush=True)
        bp.append_metrics(metrics_path, row)
        save_ckpt(latest, **state(epoch, 0, None))
        save_ckpt(os.path.join(args.log_dir, f"epoch_{epoch:02d}.pth"), model=model.state_dict(),
                  epoch=epoch)
    run_cfg["status"] = "finished"
    run_cfg["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(cfg_path, run_cfg)
    print("training finished")


if __name__ == "__main__":
    main()
