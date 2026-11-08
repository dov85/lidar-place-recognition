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
  train_step_s       time of one training step with the recipe's batch (AMP, Adam). If the batch does
                     not fit in one pass: one tuple per pass (as the training does), else 4 samples per
                     pass; train_split says which (OOM: not even that)
  epoch_min_est      steps per epoch x step time + the descriptor caches for hard-negative mining
                     (estimate on the same GPU for every method; no data loading)
  retrieval_ms       nearest neighbour of one query in a map of --db_size descriptors (CPU, numpy)
  query_total_ms     preprocessing + descriptor + retrieval: one localisation from a raw scan
comparison.csv: all of it in one table, one row per network.

  python scripts/benchmark.py --kitti_root /content/kitti --seq 08 --n 50 --out_dir <drive>/benchmark
"""
import argparse
import gc
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


# Training recipes (per epoch, 3000 training frames of sequence 00):
#   NDT-Transformer: 2 queries per step, each 1 + 2 positives + 18 negatives + 1 other = 22 clouds;
#     1500 steps; the descriptor cache for hard negatives is refreshed every 700 steps from epoch 6
#     (steps 29, 729, 1429: 3 per epoch in 15 of 20 epochs = 2.25 per epoch on average)
#   BEVPlace++ (also the own NDT network): 4 queries per step, each 1 + 1 positive + 10 negatives =
#     12 samples; 750 steps; the cache of all 3000 frames is rebuilt once per epoch
RECIPES = {"ndt": dict(tuple_size=22, tuples=2, steps=1500, caches_per_epoch=2.25),
           "bev": dict(tuple_size=12, tuples=4, steps=750, caches_per_epoch=1.0)}
TRAIN_FRAMES = 3000
EPOCHS = 20


def networks():
    """(name, model maker, input maker, recipe, preprocessing column of the representation)."""
    from lpr import bevplace as bp
    from lpr.models import ndt_net
    cells = lambda b: torch.randn(b, 1, 2000, 12)
    imgs = lambda b: torch.rand(b, 3, 201, 201)
    return [
        ("NDT-Transformer (official architecture)", lambda: ndt_net.official_twin(), cells, "ndt", "ndt_total"),
        ("own NDT, REIN (8 rotations) + Loupe",
         lambda: ndt_net.NDTNet(stn=False, rotations=8, head="loupe", feat_dim=1028),   # as the runs
         cells, "bev", "ndt_total"),
        ("own NDT, T-Net + NetVLAD (128)",
         lambda: ndt_net.NDTNet(stn=True, rotations=1, head="vlad", feat_dim=128), cells, "bev", "ndt_total"),
        ("BEVPlace++ plain", lambda: bp.build_model(None, "plain"), imgs, "bev", "bev_image"),
        ("BEVPlace++ REIN", lambda: bp.build_model(None, "rein"), imgs, "bev", "bev_image"),
    ]


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
    for name, make, inp, recipe, pre_col in networks():
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
            desc_dim = last(model(x1)).shape[-1]
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
        rec = RECIPES[recipe]
        train_mb, step_s, split = float("nan"), float("nan"), "not measured"
        if DEVICE.type == "cuda" and not a.skip_train:
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-5)
            scaler = torch.amp.GradScaler("cuda")
            xt = inp(rec["tuple_size"] * rec["tuples"]).to(DEVICE)

            def step(chunks):
                for xc in torch.chunk(xt, chunks):
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = last(model(xc)).float().pow(2).sum()
                    scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            def try_step(chunks):
                try:
                    step(chunks); sync()
                    return True
                except torch.cuda.OutOfMemoryError:
                    return False

            # 1 pass for the whole batch; else one tuple per pass with accumulated gradients (as the
            # training does); else 4 samples per pass (what the training would need on this GPU)
            n = rec["tuple_size"] * rec["tuples"]
            chunks = None
            for c in [1, rec["tuples"], max(1, n // 4)]:
                opt.zero_grad(set_to_none=True)
                gc.collect(); peak_reset()           # memory of a failed attempt is freed here
                if try_step(c):
                    chunks = c
                    break
            if chunks is None:
                split = "OOM"
                print(f"  {name}: a training step does not fit on this GPU even with 4 samples per pass")
            else:
                split = {1: "no", rec["tuples"]: "one tuple per pass"}.get(chunks, f"{n // chunks} samples per pass")
                train_mb = peak_mb()
                t0 = time.perf_counter()
                for _ in range(a.train_reps):
                    step(chunks)
                sync()
                step_s = (time.perf_counter() - t0) / a.train_reps
            del opt, xt
        dim = int(desc_dim)
        cache_s = TRAIN_FRAMES * per_scan_batch
        epoch_min = (rec["steps"] * step_s + rec["caches_per_epoch"] * cache_s) / 60
        rows.append(dict(network=name, params=n_params, rotations=rot, descriptor_dim=dim,
                         descriptor_kb=dim * 4 / 1024,
                         ms_per_scan_b1=1000 * float(np.median(ts)),
                         ms_per_scan_batch=1000 * per_scan_batch, batch=b,
                         infer_peak_mb=infer_mb, recipe=recipe,
                         train_batch=rec["tuple_size"] * rec["tuples"], train_split=split,
                         train_peak_mb=train_mb, train_step_s=step_s, steps_per_epoch=rec["steps"],
                         cache_s=cache_s, epoch_min_est=epoch_min,
                         train_hours_est=EPOCHS * epoch_min / 60, preprocessing=pre_col))
        print(f"  {name}: {rows[-1]['ms_per_scan_b1']:.1f} ms/scan, {n_params / 1e6:.2f} M params",
              flush=True)
        del model
        gc.collect(); peak_reset()
    return rows


def retrieval_ms(dim, n_db, n_q=200, seed=0):
    """Nearest neighbour of one query among n_db descriptors (brute force, as the evaluation)."""
    rng = np.random.default_rng(seed)
    db = rng.standard_normal((n_db, dim)).astype(np.float32)
    db_sq = (db ** 2).sum(1)
    qs = rng.standard_normal((n_q, dim)).astype(np.float32)
    ts = []
    for q in qs:
        t0 = time.perf_counter()
        int(np.argmin(db_sq - 2 * db @ q))
        ts.append(time.perf_counter() - t0)
    return 1000 * float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--n", type=int, default=50, help="raw scans for the preprocessing timings")
    ap.add_argument("--reps", type=int, default=20, help="batch-1 repetitions per network")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--train_reps", type=int, default=5, help="timed training steps per network")
    ap.add_argument("--db_size", type=int, default=3000, help="map size for the retrieval time")
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
    net = pd.DataFrame(network_rows(a))
    pre_mean = pre[cols].mean()
    net["retrieval_ms"] = [retrieval_ms(d, a.db_size) for d in net.descriptor_dim]
    net["query_total_ms"] = (1000 * net.preprocessing.map(pre_mean) + net.ms_per_scan_b1 + net.retrieval_ms)
    net["map_mb"] = net.descriptor_kb * a.db_size / 1024
    net = net.round(3)
    net.to_csv(os.path.join(a.out_dir, "networks.csv"), index=False)
    comp = pd.DataFrame({
        "network": net.network,
        "query: preprocessing (ms)": (1000 * net.preprocessing.map(pre_mean)).round(1),
        "query: descriptor (ms)": net.ms_per_scan_b1.round(1),
        f"query: retrieval in {a.db_size} (ms)": net.retrieval_ms.round(2),
        "query: total (ms)": net.query_total_ms.round(1),
        "inference GPU (MB)": net.infer_peak_mb.round(0),
        "train step (s)": net.train_step_s.round(3),
        "train GPU (GB)": (net.train_peak_mb / 1024).round(2),
        "train batch split": net.train_split,
        "epoch, est. (min)": net.epoch_min_est.round(1),
        f"{EPOCHS} epochs, est. (h)": net.train_hours_est.round(2),
        "descriptor (KB)": net.descriptor_kb.round(1),
        f"map of {a.db_size} (MB)": net.map_mb.round(1),
        "params (M)": (net.params / 1e6).round(2),
    })
    comp.to_csv(os.path.join(a.out_dir, "comparison.csv"), index=False)
    summary = dict(machine=machine, seq=a.seq, scans=len(pre),
                   preprocessing_seconds_per_scan=pre_sum["mean"].to_dict(),
                   networks=net.to_dict(orient="records"),
                   notes="training time estimated from timed steps on random inputs (no data loading); "
                         "cache = descriptors of the 3000 training frames for hard-negative mining")
    with open(os.path.join(a.out_dir, "benchmark.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(pre_sum.to_string())
    print(comp.T.to_string())


if __name__ == "__main__":
    main()
