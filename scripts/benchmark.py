"""Resources of the two pipelines, measured on one machine (the report's resource table).

Preprocessing, on the CPU, one process, for --n raw scans of a sequence:
  ndt_crop_ground   crop to 80 m x 80 m + Patchwork++ ground removal
  ndt_G             cell generation G (src/lpr/ndt_g.py) on the points left
  ndt_G_no_ground   G on every cropped point (ground kept)
  bev_image         BEVPlace++ image (crop, 0.4 m voxels, density image; src/lpr/bev.py)
  plus the stored size per scan (NDT cells file, BEV png).
Networks, on the GPU if there is one (random weights: time and memory do not depend on them):
  ms_per_scan_b1     descriptor of one scan (batch 1), median over repetitions
  ms_per_scan_batch  per scan, in batches of --batch (the 8-rotation networks: --batch / 8)
  infer_peak_mb      peak GPU memory of the batch-1 forward pass
  train_peak_mb      peak GPU memory of one training step with the batch of the run's recipe
                     (forward + backward, AMP as in the runs)
  params             exact number of parameters

  python scripts/benchmark.py --kitti_root /content/kitti --seq 08 --n 50 --out_dir <drive>/benchmark
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from lpr import kitti  # noqa: E402
from lpr.bev import make_bev  # noqa: E402
from lpr.experiments import BEV_DEFAULTS, NDT_DEFAULTS  # noqa: E402
from lpr.ndt_g import ndt_cells, to_official_format  # noqa: E402
from lpr.preprocess import crop, preprocess  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_KEYS = ["k", "over", "radius", "min_pts", "knn", "cov_eps"]


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def preprocessing(a):
    import cv2
    P = NDT_DEFAULTS
    g = {k: P[k] for k in G_KEYS}
    bev_args = {k: BEV_DEFAULTS[k] for k in ("half_size", "z_abs", "voxel", "res", "max_count", "ground")
                if k in BEV_DEFAULTS}
    n_total = kitti.num_frames(a.kitti_root, a.seq)
    idx = np.linspace(0, n_total - 1, min(a.n, n_total)).astype(int)
    rows = []
    for i in idx:
        scan = kitti.load_scan(kitti.scan_path(a.kitti_root, a.seq, int(i)))
        pts, t_pre = timed(preprocess, scan, P["ground"], P["half_size"], P["crop"])
        (means, covs, info), t_g = timed(lambda p: ndt_cells(p, **g), pts)
        all_pts = crop(scan, P["crop"], P["half_size"])[:, :3].astype(np.float64)
        _, t_g_all = timed(lambda p: ndt_cells(p, **g), all_pts)
        (img, _), t_bev = timed(lambda s: make_bev(s, **bev_args), scan)
        png = cv2.imencode(".png", img.astype(np.uint8))[1]
        rows.append(dict(idx=int(i), n_raw=len(scan), n_after_ground=len(pts),
                         ndt_crop_ground=t_pre, ndt_G=t_g, ndt_G_no_ground=t_g_all, bev_image=t_bev,
                         ndt_total=t_pre + t_g,
                         ndt_file_kb=to_official_format(means, covs, P["half_size"]).nbytes / 1024,
                         bev_png_kb=png.nbytes / 1024))
        print(f"  scan {i}: ndt {t_pre + t_g:.2f} s, bev {t_bev:.3f} s", flush=True)
    return rows


def networks():
    """(name, model, input maker, recipe batch for one training step)."""
    from lpr import bevplace as bp
    from lpr.models import ndt_net
    cells = lambda b: torch.randn(b, 1, 2000, 12)
    imgs = lambda b: torch.rand(b, 3, 201, 201)
    nets = [
        ("NDT-Transformer (official architecture)", lambda: ndt_net.official_twin(), cells, 2 * (1 + 2 + 18 + 1)),
        ("own NDT, REIN (8 rotations) + Loupe", lambda: ndt_net.NDTNet(stn=False, rotations=8, head="loupe",
                                                                      feat_dim=1028),   # as the runs
         cells, 4 * (1 + 1 + 10)),
        ("own NDT, T-Net + NetVLAD (128)", lambda: ndt_net.NDTNet(stn=True, rotations=1, head="vlad",
                                                                 feat_dim=128), cells, 4 * (1 + 1 + 10)),
        ("BEVPlace++ plain", lambda: bp.build_model(None, "plain"), imgs, 4 * (1 + 1 + 10)),
        ("BEVPlace++ REIN", lambda: bp.build_model(None, "rein"), imgs, 4 * (1 + 1 + 10)),
    ]
    return nets


def last(y):
    return y[-1] if isinstance(y, (tuple, list)) else y


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def peak_reset():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mb():
    return torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE.type == "cuda" else float("nan")


def network_rows(a):
    torch.backends.mha.set_fastpath_enabled(False)   # as in the evaluation of the own NDT network
    rows = []
    for name, make, inp, train_batch in networks():
        if "BEV" in name and DEVICE.type != "cuda":
            print(f"  {name}: skipped (the official REIN code needs a GPU)")
            continue
        model = make().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        rot = getattr(model, "rotations", 1) if "NDT" in name else 1
        if "REIN" in name and "BEV" in name:
            rot = len(model.rem.angles)
        model.eval()
        with torch.no_grad():
            x1 = inp(1).to(DEVICE)
            for _ in range(3):
                last(model(x1))
            peak_reset()
            ts = []
            for _ in range(a.reps):
                sync(); t0 = time.perf_counter()
                last(model(x1))
                sync(); ts.append(time.perf_counter() - t0)
            infer_mb = peak_mb()
            b = max(1, a.batch // rot)
            xb = inp(b).to(DEVICE)
            last(model(xb)); sync()
            t0 = time.perf_counter()
            for _ in range(3):
                last(model(xb))
            sync()
            per_scan_batch = (time.perf_counter() - t0) / (3 * b)
        train_mb = float("nan")
        if DEVICE.type == "cuda" and not a.skip_train:
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-5)
            peak_reset()
            try:
                xt = inp(train_batch).to(DEVICE)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = last(model(xt)).float().pow(2).sum()
                loss.backward(); opt.step(); opt.zero_grad(); sync()
                train_mb = peak_mb()
            except torch.cuda.OutOfMemoryError:
                train_mb = -1.0   # does not fit in one pass (the training splits the batch)
            del opt
        rows.append(dict(network=name, params=n_params, rotations=rot,
                         ms_per_scan_b1=1000 * float(np.median(ts)),
                         ms_per_scan_batch=1000 * per_scan_batch, batch=b,
                         infer_peak_mb=infer_mb, train_batch=train_batch, train_peak_mb=train_mb))
        print(f"  {name}: {rows[-1]['ms_per_scan_b1']:.1f} ms/scan, {n_params / 1e6:.2f} M params",
              flush=True)
        del model
        peak_reset()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--n", type=int, default=50, help="raw scans for the preprocessing timings")
    ap.add_argument("--reps", type=int, default=20, help="batch-1 repetitions per network")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    import pandas as pd
    machine = dict(cpu=platform.processor() or platform.machine(), cores=os.cpu_count(),
                   gpu=torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "none",
                   torch=torch.__version__)
    print("machine:", machine)
    print("preprocessing (CPU, one process)")
    pre = pd.DataFrame(preprocessing(a))
    pre.to_csv(os.path.join(a.out_dir, "preprocessing_per_scan.csv"), index=False)
    cols = ["ndt_crop_ground", "ndt_G", "ndt_total", "ndt_G_no_ground", "bev_image", "ndt_file_kb", "bev_png_kb",
            "n_raw", "n_after_ground"]
    pre_sum = pre[cols].agg(["mean", "median"]).T.round(4)
    print("networks on", DEVICE)
    net = pd.DataFrame(network_rows(a)).round(2)
    net.to_csv(os.path.join(a.out_dir, "networks.csv"), index=False)
    summary = dict(machine=machine, seq=a.seq, scans=len(pre),
                   preprocessing_seconds_per_scan=pre_sum["mean"].to_dict(),
                   networks=net.to_dict(orient="records"))
    with open(os.path.join(a.out_dir, "benchmark.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(pre_sum.to_string())
    print(net.to_string(index=False))


if __name__ == "__main__":
    main()
