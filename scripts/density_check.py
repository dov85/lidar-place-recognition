"""Why do some runs fail when the map has 64 beams and the queries 16 ("query" mode of the
sparsity evaluation)? Descriptors of the same scans with 64 beams and with fewer beams.

For n scans spread over a sequence (about 10 m apart), from the run's 64-beam variant and from
each sparse variant (folders next to --data_dir, as in the sparsity evaluation):
  self_top1     % of sparse descriptors whose nearest 64-beam descriptor (among the n) is the
                same scan: density robustness without the rest of the map
  self_dist     median distance between the 64-beam and the sparse descriptor of a scan
  other_dist    median distance between 64-beam descriptors of different scans
  ratio         self_dist / other_dist: < 1 the scan stays closer to itself than to others
  spread        median pairwise distance among the sparse descriptors / among the 64-beam ones
                (well below 1: the sparse descriptors collapse towards one point)
  shift         distance between the mean sparse and the mean 64-beam descriptor / other_dist
  hub_share     % of sparse queries whose nearest 64-beam descriptor is the single most chosen one
  norm_64, norm_sparse  median descriptor norm (L2-normalised heads: 1)

  python scripts/density_check.py --method bev --run_dir <drive>/runs/bev/<run> \
      --data_dir /content/bev/<variant> --kitti_root /content/kitti --seq 08 --out check.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "NDT-Transformer"))
from lpr import evals  # noqa: E402
from lpr import experiments as ex  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bev_describer(ckpt, official=None):
    from lpr import bevplace as bp
    model = bp.build_model(official, bp.run_arch(ckpt)).to(DEVICE)
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck.get("model", ck.get("state_dict", ck)))

    def describe(folder, seq, idx):
        paths = bp.image_paths(folder, seq)
        return bp.infer(model, bp.InferDataset([paths[i] for i in idx]), DEVICE, batch=16)
    return describe


def ndt_describer(ckpt):
    from lpr.models import ndt_net
    from lpr.myndt import load_cells
    mine = ndt_net.run_params(ckpt)
    run_cfg = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "config.json")
    cov = False
    if os.path.exists(run_cfg):
        cov = bool(json.load(open(run_cfg)).get("params", {}).get("cov_transform", False))
    if mine is not None:
        net = ndt_net.build(dict(mine, cov_transform=cov))
        torch.backends.mha.set_fastpath_enabled(False)
        batch = max(1, 32 // net.rotations)
    else:
        import models.NDTNetVlad as official_model
        net = official_model.NDTNetVlad(num_points=2000, output_dim=256, emb_dims=1024,
                                        layer_number=3)
        net.Transformer.convariance_transform = cov
        batch = 32
    net = net.to(DEVICE)
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    net.load_state_dict(ck.get("model", ck.get("state_dict", ck)))
    net.eval()

    @torch.no_grad()
    def describe(folder, seq, idx):
        out = []
        for s in range(0, len(idx), batch):
            x = np.stack([load_cells(os.path.join(folder, seq, f"{i:06d}.bin"))
                          for i in idx[s:s + batch]]).astype(np.float32)
            y = net(torch.from_numpy(x).unsqueeze(1).to(DEVICE))
            out.append((y[-1] if isinstance(y, tuple) else y).float().cpu().numpy())
        return np.concatenate(out)
    return describe


def pairwise(a, b):
    d = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.clip(d, 0, None))


def metrics(d64, ds):
    n = len(d64)
    off = ~np.eye(n, dtype=bool)
    p64 = pairwise(d64, d64)
    other = float(np.median(p64[off]))
    cross = pairwise(ds, d64)                 # sparse query i -> 64-beam scan j
    self_d = float(np.median(np.diag(cross)))
    nn = cross.argmin(1)
    counts = np.bincount(nn, minlength=n)
    return {
        "self_top1": round(100.0 * float(np.mean(nn == np.arange(n))), 1),
        "self_dist": round(self_d, 4),
        "other_dist": round(other, 4),
        "ratio": round(self_d / other, 3),
        "spread": round(float(np.median(pairwise(ds, ds)[off])) / other, 3),
        "shift": round(float(np.linalg.norm(ds.mean(0) - d64.mean(0))) / other, 3),
        "hub_share": round(100.0 * counts.max() / n, 1),
        "norm_64": round(float(np.median(np.linalg.norm(d64, axis=1))), 3),
        "norm_sparse": round(float(np.median(np.linalg.norm(ds, axis=1))), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ndt", "bev", "myndt"])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="last", help="last | epoch_20.pth | a path")
    ap.add_argument("--data_dir", required=True, help="the run's 64-beam variant on this machine")
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--n", type=int, default=400, help="scans, spread evenly over the sequence")
    ap.add_argument("--beams", type=int, nargs="+", default=[32, 16])
    ap.add_argument("--official", default=None, help="third_party/BEVPlace2 (bev)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ckpt = evals.resolve_ckpt(a.run_dir, a.ckpt)
    data = "bev" if a.method == "bev" else "ndt"
    base = os.path.basename(os.path.normpath(a.data_dir))
    params = json.load(open(os.path.join(a.data_dir, "config.json")))["params"]
    ext = ".png" if data == "bev" else ".bin"
    total = len([f for f in os.listdir(os.path.join(a.data_dir, a.seq)) if f.endswith(ext)])
    idx = np.unique(np.linspace(0, total - 1, min(a.n, total)).astype(int))
    describe = bev_describer(ckpt, a.official) if data == "bev" else ndt_describer(ckpt)
    d64 = describe(a.data_dir, a.seq, idx)
    rows = []
    for b in a.beams:
        name, _ = ex.sparse_variant(data, base, params, b)
        sdir = os.path.join(os.path.dirname(os.path.normpath(a.data_dir)), name)
        if not os.path.isdir(os.path.join(sdir, a.seq)):
            print(f"{b} beams: {sdir}/{a.seq} not found (skipped)")
            continue
        r = {"run": os.path.basename(os.path.normpath(a.run_dir)), "method": a.method,
             "checkpoint": evals.ckpt_name(ckpt), "seq": a.seq, "beams": b, "n": len(idx),
             **metrics(d64, describe(sdir, a.seq, idx))}
        rows.append(r)
        print(json.dumps(r))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
