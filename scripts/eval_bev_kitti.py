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

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from lpr import bevplace as bp  # noqa: E402
from lpr import kitti  # noqa: E402
from lpr.evaluation import recall  # noqa: E402

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
    args = ap.parse_args()

    model = bp.build_model(args.official).to(DEVICE)
    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck.get("model", ck.get("state_dict", ck)))

    paths = bp.image_paths(args.bev_dir, args.seq)
    n = len(paths)
    t0 = time.time()
    desc = bp.infer(model, bp.InferDataset(paths), DEVICE, batch=args.batch, workers=args.workers)
    sec = (time.time() - t0) / n
    xyz = kitti.positions(kitti.load_poses(args.kitti_root, args.seq))[:n]
    db, q = kitti.split_indices(args.seq, n)
    r = recall(desc[db], desc[q], xyz[db], xyz[q], threshold=kitti.MATCH_THRESHOLD_M)
    top1 = r.pop("top1_indices")
    r.update(seq=args.seq, ckpt=args.ckpt, method="bev", network_seconds_per_scan=sec)
    print(json.dumps(r, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f, indent=1)
        np.savez(os.path.splitext(args.out)[0] + "_matches.npz", query=q, top1=top1,
                 descriptors=desc)


if __name__ == "__main__":
    main()
