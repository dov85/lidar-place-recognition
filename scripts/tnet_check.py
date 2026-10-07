"""What did the T-Net of an NDT-Transformer checkpoint learn?

Takes n scans (evenly spaced) of one sequence, rotates their NDT cells about the vertical
axis by each angle (means: R m, covariances: R S R^T) and records, per angle:
  T_dev_I        ||T - I||_F, mean over scans (how far the T-Net is from the identity)
  T_det          determinant of T, mean
  T_static       ||T(a) - T(0)||_F / ||T(0)||_F    (0: T does not react to the rotation)
  T_follow       ||T(a) - R T(0)||_F / ||T(0)||_F  (0: T undoes the rotation exactly)
  T_yaw_deg      rotation about z in the polar part of T(a) T(0)^-1, median
                 (the angle the T-Net compensates; ideal = the input angle)
  desc_drift     ||g(a) - g(0)|| / median distance between descriptors of different scans
                 (0: rotation invariant; ~1: as different as another place)
  self_top1      share of rotated scans whose nearest unrotated descriptor is themselves
Writes summary.csv, tnet_T0.json (mean T at 0 deg) and tnet_angles.png into --out_dir.

Example:
  python scripts/tnet_check.py --ckpt <run>/epoch_20.pth --ndt_dir /content/ndt/<variant> --seq 08 \
      --out_dir <run>/eval/tnet/epoch_20
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "NDT-Transformer"))
import config as cfg  # noqa: E402
import loading_pointclouds as lp  # noqa: E402
import models.NDTNetVlad as official_model  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def rotate_cells(x, R):
    """x: (n, k, 12) rows [mean, cov(9)] -> means R m, covariances R S R^T."""
    m = x[..., :3] @ R.T
    S = x[..., 3:].reshape(*x.shape[:2], 3, 3)
    S = R @ S @ R.T
    return np.concatenate([m, S.reshape(*x.shape[:2], 9)], axis=-1)


@torch.no_grad()
def forward(model, x, batch=25):
    Ts, gs = [], []
    for s in range(0, len(x), batch):
        t = torch.from_numpy(x[s:s + batch]).float().unsqueeze(1).to(DEVICE)
        Ts.append(model.Transformer.point_trans(t[..., :3]).cpu().numpy())
        gs.append(model(t).cpu().numpy())
    return np.concatenate(Ts), np.concatenate(gs)


def polar_yaw(M):
    W, _, Vt = np.linalg.svd(M)
    U = W @ Vt
    return np.degrees(np.arctan2(U[1, 0], U[0, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ndt_dir", required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--n_scans", type=int, default=200)
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 15, 30, 45, 60, 90, 120, 150, 180])
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    model = official_model.NDTNetVlad(num_points=2000, output_dim=256, emb_dims=1024,
                                      layer_number=3).to(DEVICE)
    ck = torch.load(a.ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck.get("model", ck.get("state_dict", ck)))
    model.eval()
    run_cfg = os.path.join(os.path.dirname(os.path.abspath(a.ckpt)), "config.json")
    cov = bool(json.load(open(run_cfg)).get("params", {}).get("cov_transform", False)) \
        if os.path.exists(run_cfg) else False
    model.Transformer.convariance_transform = cov

    cfg.DATASET_FOLDER = a.ndt_dir
    n_all = len(glob.glob(os.path.join(a.ndt_dir, a.seq, "*.bin")))
    idx = np.linspace(0, n_all - 1, min(a.n_scans, n_all)).astype(int)
    x0 = lp.load_pc_files([os.path.join(a.seq, f"{i:06d}.bin") for i in idx]).astype(np.float64)

    T0, g0 = forward(model, x0)
    d_other = np.linalg.norm(g0[:, None] - g0[None], axis=-1)
    scale = np.median(d_other[~np.eye(len(g0), dtype=bool)])
    I = np.eye(3)
    rows = []
    for ang in a.angles:
        R = rz(ang)
        T, g = forward(model, rotate_cells(x0, R)) if ang != 0 else (T0, g0)
        n0 = np.linalg.norm(T0, axis=(1, 2))
        d = np.linalg.norm(g[:, None] - g0[None], axis=-1)
        yaws = [polar_yaw(T[i] @ np.linalg.inv(T0[i])) for i in range(len(T))]
        rows.append({
            "angle": ang,
            "T_dev_I": float(np.mean(np.linalg.norm(T - I, axis=(1, 2)))),
            "T_det": float(np.mean(np.linalg.det(T))),
            "T_static": float(np.mean(np.linalg.norm(T - T0, axis=(1, 2)) / n0)),
            "T_follow": float(np.mean(np.linalg.norm(T - R @ T0, axis=(1, 2)) / n0)),
            "T_yaw_deg": float(np.median(yaws)),
            "desc_drift": float(np.mean(np.diag(d)) / scale),
            "self_top1": float(np.mean(d.argmin(axis=1) == np.arange(len(g)))),
        })
    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(a.out_dir, "summary.csv"), index=False)
    json.dump({"seq": a.seq, "n_scans": int(len(idx)), "cov_transform": cov,
               "T0_mean": T0.mean(0).round(4).tolist(), "T0_std": T0.std(0).round(4).tolist()},
              open(os.path.join(a.out_dir, "tnet_T0.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(df.angle, df.T_static, "o-", label="T(a) vs T(0)  (0 = ignores rotation)")
    ax[0].plot(df.angle, df.T_follow, "o-", label="T(a) vs R T(0)  (0 = undoes rotation)")
    ax[0].set_title("T-Net response"); ax[0].legend(fontsize=7)
    ax[1].plot(df.angle, df.angle, "k--", lw=1, label="ideal")
    ax[1].plot(df.angle, df.T_yaw_deg, "o-", label="compensated by T")
    ax[1].set_title("yaw compensated (deg)"); ax[1].legend(fontsize=7)
    ax[2].plot(df.angle, df.desc_drift, "o-", label="descriptor drift")
    ax[2].plot(df.angle, df.self_top1, "o-", label="finds itself (top-1)")
    ax[2].set_title("descriptor"); ax[2].legend(fontsize=7)
    for x in ax:
        x.set_xlabel("input rotation (deg)")
    fig.suptitle(f"{os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))}  seq {a.seq}, "
                 f"{len(idx)} scans, cov_transform={cov}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "tnet_angles.png"), dpi=120)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
