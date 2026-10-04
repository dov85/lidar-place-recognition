"""Evaluate an NDT-Transformer checkpoint on one KITTI sequence.

Example:
  python scripts/eval_ndt_kitti.py --ckpt runs/ndt_00/epoch_20.pth --ndt_dir /content/ndt \
      --kitti_root /content/kitti --seq 00 --out runs/ndt_00/eval_00.json
"""
import argparse
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ndt_dir", required=True)
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--split", type=int, default=None, help="override the split point")
    ap.add_argument("--gap", type=int, default=kitti.QUERY_GAP)
    args = ap.parse_args()
    cfg.DATASET_FOLDER = args.ndt_dir

    model = official_model.NDTNetVlad(num_points=2000, output_dim=256, emb_dims=1024,
                                      layer_number=3).to(DEVICE)
    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck.get("model", ck.get("state_dict", ck)))
    model.eval()

    n = kitti.num_frames(args.kitti_root, args.seq)
    files = [os.path.join(args.seq, f"{i:06d}.bin") for i in range(n)]
    t0 = time.time()
    desc = []
    with torch.no_grad():
        for s in range(0, n, args.batch):
            x = torch.from_numpy(lp.load_pc_files(files[s:s + args.batch])).float()
            desc.append(model(x.unsqueeze(1).to(DEVICE)).cpu().numpy())
    desc = np.concatenate(desc)
    sec = (time.time() - t0) / n

    xyz = kitti.positions(kitti.load_poses(args.kitti_root, args.seq))[:n]
    if args.split is None:
        db, q = kitti.split_indices(args.seq, n)
    else:
        db, q = np.arange(args.split), np.arange(args.split + args.gap, n)
    r = recall(desc[db], desc[q], xyz[db], xyz[q], threshold=kitti.MATCH_THRESHOLD_M)
    top1 = r.pop("top1_indices")
    r.update(seq=args.seq, ckpt=args.ckpt, network_seconds_per_scan=sec)
    print(json.dumps(r, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f, indent=1)
        np.savez(os.path.splitext(args.out)[0] + "_matches.npz", query=q, top1=top1,
                 descriptors=desc)


if __name__ == "__main__":
    main()
