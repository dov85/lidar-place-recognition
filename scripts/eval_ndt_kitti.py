"""Evaluate an NDT-Transformer checkpoint on one KITTI sequence.

Official network (runs of train_ndt_kitti.py), or the own network (src/lpr/models/ndt_net.py,
runs of train_myndt_kitti.py: config.json next to the checkpoint with method "myndt", whose
network parameters are used).

Example:
  python scripts/eval_ndt_kitti.py --ckpt runs/ndt_00/epoch_20.pth --ndt_dir /content/ndt \
      --kitti_root /content/kitti --seq 00 --out runs/ndt_00/eval_00.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "NDT-Transformer"))
import config as cfg  # noqa: E402
import loading_pointclouds as lp  # noqa: E402
import models.NDTNetVlad as official_model  # noqa: E402

from lpr import kitti  # noqa: E402
from lpr.evaluation import recall  # noqa: E402
from lpr.models import ndt_net  # noqa: E402
from lpr.rotation import rotate_cells  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ndt_dir", required=True)
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=32,
                    help="scans per forward pass; divided by the number of rotations of an own "
                         "network (memory only)")
    ap.add_argument("--split", type=int, default=None, help="override the split point")
    ap.add_argument("--gap", type=int, default=kitti.QUERY_GAP)
    ap.add_argument("--yaw_db", default=None,
                    help="compass test: .npy with one angle (deg) per frame; database frames are "
                         "rotated about z by it")
    ap.add_argument("--yaw_q", nargs="+", default=None,
                    help="compass test: .npy files with one angle per frame for the queries; one "
                         "result per file, written to the matching --outs entry")
    ap.add_argument("--outs", nargs="+", default=None)
    ap.add_argument("--query_dir", default=None,
                    help="sparsity test: the queries come from this variant folder (e.g. 16 beams), "
                         "the database from --ndt_dir")
    ap.add_argument("--cov_transform", choices=["auto", "on", "off"], default="auto",
                    help="auto: as in config.json next to the checkpoint (the run's setting)")
    args = ap.parse_args()
    cfg.DATASET_FOLDER = args.ndt_dir

    mine = ndt_net.run_params(args.ckpt)
    cov = args.cov_transform == "on"
    if args.cov_transform == "auto":
        run_cfg = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config.json")
        if os.path.exists(run_cfg):
            cov = bool(json.load(open(run_cfg)).get("params", {}).get("cov_transform", False))
    if mine is not None:
        net = ndt_net.build(dict(mine, cov_transform=cov))
        # every scan goes through the encoder once per rotation; PyTorch's inference fast path of
        # the transformer builds the full attention matrices (8 x 32 scans x 4 heads x 2000^2 =
        # 15 GB), so use the memory-efficient attention and fewer scans per pass
        torch.backends.mha.set_fastpath_enabled(False)
        args.batch = max(1, args.batch // net.rotations)
        print("own NDT network:", {k: mine.get(k) for k in ndt_net.MODEL_KEYS})
    else:
        net = official_model.NDTNetVlad(num_points=2000, output_dim=256, emb_dims=1024,
                                        layer_number=3)
        net.Transformer.convariance_transform = cov
    net = net.to(DEVICE)
    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    net.load_state_dict(ck.get("model", ck.get("state_dict", ck)))
    net.eval()

    def model(x):   # descriptors of (B, 1, N, 12)
        out = net(x)
        return out[-1] if isinstance(out, tuple) else out

    n = len(glob.glob(os.path.join(args.ndt_dir, args.seq, "*.bin")))
    files = [os.path.join(args.seq, f"{i:06d}.bin") for i in range(n)]
    if args.yaw_q:
        return compass(args, model, files, n, cov)
    t0 = time.time()

    def describe(folder, names):
        cfg.DATASET_FOLDER = folder
        with torch.no_grad():
            return in_batches(lambda sl: model(torch.from_numpy(lp.load_pc_files(names[sl])).float()
                                               .unsqueeze(1).to(DEVICE)).cpu().numpy(),
                              len(names), args)

    desc = describe(args.ndt_dir, files)
    sec = (time.time() - t0) / n

    xyz = kitti.positions(kitti.load_poses(args.kitti_root, args.seq))[:n]
    if args.split is None:
        db, q = kitti.split_indices(args.seq, n)
    else:
        db, q = np.arange(args.split), np.arange(args.split + args.gap, n)
    if args.query_dir:   # queries from another variant (same scans, e.g. fewer beams)
        desc[q] = describe(args.query_dir, [files[i] for i in q])
    r = recall(desc[db], desc[q], xyz[db], xyz[q], threshold=kitti.MATCH_THRESHOLD_M)
    top1 = r.pop("top1_indices")
    r.update(seq=args.seq, ckpt=args.ckpt, cov_transform=cov, network_seconds_per_scan=sec,
             db_dir=args.ndt_dir, query_dir=args.query_dir or args.ndt_dir)
    print(json.dumps(r, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f, indent=1)
        np.savez(os.path.splitext(args.out)[0] + "_matches.npz", query=q, top1=top1,
                 descriptors=desc)


def in_batches(fn, n, args):
    """fn(slice) -> descriptors of those items, over range(n) in batches of args.batch. If a
    batch does not fit in GPU memory, the batch size is halved (and kept) and the batch retried."""
    out, s = [], 0
    while s < n:
        try:
            out.append(fn(slice(s, s + args.batch)))
            s += args.batch
        except torch.cuda.OutOfMemoryError:
            if args.batch == 1:
                raise
            torch.cuda.empty_cache()
            args.batch = max(1, args.batch // 2)
            print(f"  out of GPU memory: {args.batch} scans per pass from now on", flush=True)
    return np.concatenate(out)


@torch.no_grad()
def descriptors(model, files, idx, angles, args):
    def fn(sl):
        b = idx[sl]
        x = lp.load_pc_files([files[i] for i in b])
        if angles is not None:
            x = rotate_cells(x, angles[b])
        return model(torch.from_numpy(x).float().unsqueeze(1).to(DEVICE)).cpu().numpy()
    return in_batches(fn, len(idx), args)


def compass(args, model, files, n, cov):
    """Database frames rotated by --yaw_db, queries by each --yaw_q; one result per --outs."""
    xyz = kitti.positions(kitti.load_poses(args.kitti_root, args.seq))[:n]
    db, q = kitti.split_indices(args.seq, n)
    yaw_db = np.load(args.yaw_db) if args.yaw_db else None
    d_db = descriptors(model, files, db, yaw_db, args)
    for yq, out in zip(args.yaw_q, args.outs):
        d_q = descriptors(model, files, q, np.load(yq), args)
        r = recall(d_db, d_q, xyz[db], xyz[q], threshold=kitti.MATCH_THRESHOLD_M)
        top1 = r.pop("top1_indices")
        r.update(seq=args.seq, ckpt=args.ckpt, cov_transform=cov, yaw_db=args.yaw_db, yaw_q=yq)
        with open(out, "w") as f:
            json.dump(r, f, indent=1)
        np.savez(os.path.splitext(out)[0] + "_matches.npz", query=q, top1=top1)
        print(os.path.basename(os.path.dirname(out)), f"R@1 {r['R@1']:.2f}", flush=True)


if __name__ == "__main__":
    main()
