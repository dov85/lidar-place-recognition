"""Evaluate a BEVPlace++ checkpoint on one KITTI sequence (same protocol and outputs as
eval_ndt_kitti.py: <out>.json and <out>_matches.npz for heading_analysis.py).

Accepts our checkpoints (epoch_XX.pth, key "model") and the official ones
(model_best.pth.tar / checkpoint.pth.tar, key "state_dict").

Example:
  python scripts/eval_bev_kitti.py --ckpt third_party/BEVPlace2/runs/Aug08_10-17-29/model_best.pth.tar \
      --bev_dir /content/bev/<variant> --kitti_root /content/kitti --seq 08 --out /content/eval_08.json
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from lpr import bevplace as bp  # noqa: E402
from lpr import kitti  # noqa: E402
from lpr.evaluation import recall  # noqa: E402
from lpr.rotation import rotate_bev  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bev_dir", required=True)
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--out", default=None)
    ap.add_argument("--official", default=None, help="third_party/BEVPlace2")
    ap.add_argument("--batch", type=int, default=32, help="images per forward pass (memory only)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--yaw_db", default=None,
                    help="compass test: .npy with one angle (deg) per frame; database frames are "
                         "rotated about z by it")
    ap.add_argument("--yaw_q", nargs="+", default=None,
                    help="compass test: .npy files with one angle per frame for the queries; one "
                         "result per file, written to the matching --outs entry")
    ap.add_argument("--outs", nargs="+", default=None)
    ap.add_argument("--query_dir", default=None,
                    help="sparsity test: the queries come from this variant folder (e.g. 16 beams), "
                         "the database from --bev_dir")
    ap.add_argument("--split", type=int, default=None,
                    help="override the split point (database = frames before it), e.g. 3000 for 08 "
                         "as in Table II of the BEVPlace++ paper (the official code uses 1000)")
    ap.add_argument("--gap", type=int, default=kitti.QUERY_GAP)
    ap.add_argument("--arch", default="auto", choices=["auto", *bp.ARCHS],
                    help="auto: from the config.json of the checkpoint's run (official: rein)")
    args = ap.parse_args()

    arch = bp.run_arch(args.ckpt) if args.arch == "auto" else args.arch
    print("network:", arch)
    model = bp.build_model(args.official, arch).to(DEVICE)
    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck.get("model", ck.get("state_dict", ck)))

    paths = bp.image_paths(args.bev_dir, args.seq)
    n = len(paths)
    if args.yaw_q:
        return compass(args, model, paths, n)
    t0 = time.time()
    desc = bp.infer(model, bp.InferDataset(paths), DEVICE, batch=args.batch, workers=args.workers)
    sec = (time.time() - t0) / n
    xyz = kitti.positions(kitti.load_poses(args.kitti_root, args.seq))[:n]
    db, q = split(args, n)
    if args.query_dir:   # queries from another variant (same scans, e.g. fewer beams)
        qpaths = bp.image_paths(args.query_dir, args.seq)
        desc[q] = bp.infer(model, bp.InferDataset([qpaths[i] for i in q]), DEVICE,
                           batch=args.batch, workers=args.workers)
    r = recall(desc[db], desc[q], xyz[db], xyz[q], threshold=kitti.MATCH_THRESHOLD_M)
    top1 = r.pop("top1_indices")
    r.update(seq=args.seq, ckpt=args.ckpt, method="bev", network_seconds_per_scan=sec,
             split=int(db[-1]) + 1,
             db_dir=args.bev_dir, query_dir=args.query_dir or args.bev_dir)
    print(json.dumps(r, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f, indent=1)
        np.savez(os.path.splitext(args.out)[0] + "_matches.npz", query=q, top1=top1,
                 descriptors=desc)


def split(args, n):
    """Database and query frames: the protocol's split point, or --split / --gap."""
    if args.split is None:
        return kitti.split_indices(args.seq, n)
    return np.arange(args.split), np.arange(args.split + args.gap, n)


class RotatedImages(bp.InferDataset):
    """Evaluation images rotated by one angle (deg) each, like the points rotated about z."""

    def __init__(self, img_paths, angles):
        super().__init__(img_paths)
        self.angles = angles

    def __getitem__(self, index):
        img = rotate_bev(cv2.imread(self.imgs_path[index], 0), self.angles[index])
        img = img.astype(np.float32) / 256
        return img[np.newaxis].repeat(3, 0), index


def compass(args, model, paths, n):
    """Database frames rotated by --yaw_db, queries by each --yaw_q; one result per --outs."""
    xyz = kitti.positions(kitti.load_poses(args.kitti_root, args.seq))[:n]
    db, q = split(args, n)
    yaw_db = np.load(args.yaw_db) if args.yaw_db else np.zeros(n)

    def desc(idx, angles):
        ds = RotatedImages([paths[i] for i in idx], angles[idx])
        return bp.infer(model, ds, DEVICE, batch=args.batch, workers=args.workers)

    d_db = desc(db, yaw_db)
    for yq, out in zip(args.yaw_q, args.outs):
        d_q = desc(q, np.load(yq))
        r = recall(d_db, d_q, xyz[db], xyz[q], threshold=kitti.MATCH_THRESHOLD_M)
        top1 = r.pop("top1_indices")
        r.update(seq=args.seq, ckpt=args.ckpt, method="bev", yaw_db=args.yaw_db, yaw_q=yq)
        with open(out, "w") as f:
            json.dump(r, f, indent=1)
        np.savez(os.path.splitext(out)[0] + "_matches.npz", query=q, top1=top1)
        print(os.path.basename(os.path.dirname(out)), f"R@1 {r['R@1']:.2f}", flush=True)


if __name__ == "__main__":
    main()
