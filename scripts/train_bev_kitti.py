"""Train BEVPlace++ on KITTI with the official network, loss and sampling.

From the official repository (third_party/BEVPlace2): the REIN network; the training
set, hard mining, loss and NetVLAD initialisation are reproduced in src/lpr/bevplace.py.
New here: checkpoint/resume that survives Colab disconnects (also inside an epoch),
per-epoch evaluation on the training sequence with the same protocol as NDT-Transformer,
and the run bookkeeping (config.json, metrics.csv) shared with train_ndt_kitti.py.

Official hyper-parameters: frames 0-2999 of KITTI 00, positives < 5 m, negatives > 7 m,
10 negatives, batch 4, triplet margin 0.3, Adam lr 1e-4, 40 epochs, random rotation.

Example:
  python scripts/train_bev_kitti.py --bev_dir /content/bev/<variant> --kitti_root /content/kitti \
      --bev_variant <variant> --log_dir /content/drive/MyDrive/lpr/runs/bev/<run>
"""
import argparse
import csv
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
from lpr import kitti  # noqa: E402
from lpr.evaluation import recall  # noqa: E402
from lpr.experiments import BEV_TRAIN_DEFAULTS, full_params, git_commit, run_hash, write_json  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse():
    d = BEV_TRAIN_DEFAULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--bev_dir", required=True, help="<bev_dir>/<seq>/<idx>.png")
    ap.add_argument("--kitti_root", required=True, help="for poses")
    ap.add_argument("--log_dir", required=True)
    ap.add_argument("--bev_variant", default="")
    ap.add_argument("--official", default=None, help="third_party/BEVPlace2")
    for k in ("seq", "train_frames", "pos_thr", "neg_thr", "epochs", "lr", "batch", "negatives",
              "margin", "seed"):
        ap.add_argument(f"--{k}", type=type(d[k]), default=d[k])
    ap.add_argument("--no_rot_aug", action="store_true", help="disable the random rotation")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=1, help="epochs; 0 = never")
    ap.add_argument("--eval_batch", type=int, default=32,
                    help="images per forward pass when computing descriptors (memory only)")
    ap.add_argument("--ckpt_every", type=int, default=150, help="iterations")
    ap.add_argument("--max_iters", type=int, default=0, help="per epoch, testing only")
    ap.add_argument("--n_cluster_desc", type=int, default=10000, help="testing only")
    ap.add_argument("--split_forward", action="store_true",
                    help="one tuple per forward pass from the start (otherwise only after an OOM)")
    return ap.parse_args()


def save_ckpt(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def evaluate(model, args, paths, xyz):
    t0 = time.time()
    desc = bp.infer(model, bp.InferDataset(paths), DEVICE, batch=args.eval_batch,
                    workers=args.workers)
    db, q = kitti.split_indices(args.seq, len(paths))
    r = recall(desc[db], desc[q], xyz[db], xyz[q], threshold=kitti.MATCH_THRESHOLD_M)
    r.pop("top1_indices")
    r["descriptor_seconds_per_scan"] = (time.time() - t0) / len(paths)
    return r


def main():
    args = parse()
    args.rot_aug = not args.no_rot_aug
    os.makedirs(args.log_dir, exist_ok=True)
    params = full_params(BEV_TRAIN_DEFAULTS, {k: getattr(args, k) for k in BEV_TRAIN_DEFAULTS})
    phash = run_hash(params, args.bev_variant, "bev")
    cfg_path = os.path.join(args.log_dir, "config.json")
    old = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
    if old is not None and old.get("params_hash") != phash:
        raise SystemExit(f"{args.log_dir} already holds a run with different parameters "
                         f"(hash {old.get('params_hash')} != {phash}). Use a new log_dir / run name.")
    run_cfg = old or {
        "run": os.path.basename(os.path.normpath(args.log_dir)),
        "method": "bev",
        "bev_variant": args.bev_variant,
        "params": params,
        "params_hash": phash,
        "paths": {"bev_dir": args.bev_dir},
        "code_commit": git_commit(ROOT),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sessions": 0,
    }
    run_cfg["status"] = "running"
    run_cfg["sessions"] = run_cfg.get("sessions", 0) + 1
    write_json(cfg_path, run_cfg)
    var_cfg = os.path.join(args.bev_dir, "config.json")
    if os.path.exists(var_cfg):
        write_json(os.path.join(args.log_dir, "bev_config.json"), json.load(open(var_cfg)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    paths = bp.image_paths(args.bev_dir, args.seq)
    poses = kitti.load_poses(args.kitti_root, args.seq)[:len(paths)]
    xyz = kitti.positions(poses)
    train_set = bp.TrainingDataset(paths, poses.reshape(len(poses), 12), args.train_frames,
                                   args.pos_thr, args.neg_thr, args.negatives, args.rot_aug)
    model = bp.build_model(args.official).to(DEVICE)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    latest = os.path.join(args.log_dir, "latest.pth")
    epoch, start_iter, perm = 0, 0, None
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        epoch, start_iter, perm = ck["epoch"], ck["iter"], ck["perm"]
        np.random.seed(args.seed + 1000 * epoch + start_iter)
        print(f"resumed from {latest}: epoch {epoch}, iteration {start_iter}")
    else:
        print("NetVLAD initialisation: k-means on local features of the training images")
        t0 = time.time()
        centroids, descs = bp.get_clusters(model, train_set, DEVICE, workers=args.workers,
                                           n_descriptors=args.n_cluster_desc, seed=args.seed)
        np.savez(os.path.join(args.log_dir, "netvlad_init.npz"), centroids=centroids)
        model.pooling.init_params(centroids, descs)
        model = model.to(DEVICE)
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        print(f"  done in {time.time() - t0:.0f} s")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device {DEVICE}, parameters {n_params / 1e6:.2f} M, training frames {len(train_set)}, "
          f"images {len(paths)}")

    metrics_path = os.path.join(args.log_dir, "metrics.csv")
    split = [args.split_forward]
    while epoch < args.epochs:
        t_epoch = time.time()
        if perm is None:
            perm = np.random.permutation(len(train_set))
        batches = [perm[i:i + args.batch].tolist() for i in range(0, len(perm), args.batch)]
        if args.max_iters:
            batches = batches[:args.max_iters]
        print(f"epoch {epoch + 1}: descriptor cache for hard mining", flush=True)
        t0 = time.time()
        bp.build_cache(model, train_set, DEVICE, batch=args.batch, workers=args.workers)
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
                save_ckpt(latest, model=model.state_dict(), optimizer=opt.state_dict(),
                          epoch=epoch, iter=it, perm=perm)

        losses = bp.train_batches(model, train_set, batches, opt, DEVICE, args.batch, args.margin,
                                  args.workers, log, on_iter, start=start_iter, split=split)
        epoch += 1
        start_iter, perm = 0, None
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"),
               "epoch_minutes": round((time.time() - t_epoch) / 60, 2),
               "cache_minutes": round(cache_min, 2), "split_forward": split[0]}
        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
            r = evaluate(model, args, paths, xyz)
            row.update({k: r[k] for k in ("R@1", "R@5", "R@1%", "n_queries_with_positive")})
        print("EPOCH SUMMARY", json.dumps(row), flush=True)
        new = not os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        save_ckpt(latest, model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch,
                  iter=0, perm=None)
        save_ckpt(os.path.join(args.log_dir, f"epoch_{epoch:02d}.pth"), model=model.state_dict(),
                  epoch=epoch)
    run_cfg["status"] = "finished"
    run_cfg["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(cfg_path, run_cfg)
    print("training finished")


if __name__ == "__main__":
    main()
