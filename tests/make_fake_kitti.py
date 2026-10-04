"""Build a tiny fake KITTI (same zip layout) to test the pipeline end to end.

The car drives 1 m per frame along a synthetic street and then drives back
along the same street (a revisit in the opposite direction).
"""
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from synthetic import synthetic_scan  # noqa: E402


def world(length=300, seed=0):
    rng = np.random.default_rng(seed)
    tiles = []
    for i, x0 in enumerate(range(-60, length + 60, 100)):
        s = synthetic_scan(seed=seed * 1000 + i, n=400_000)[:, :3]
        s = s[(s[:, 0] > -50) & (s[:, 0] <= 50)]
        tiles.append(s + [x0 + 50, 0, 0])
    return np.concatenate(tiles), rng


def main(out_dir, n_frames=240):
    os.makedirs(out_dir, exist_ok=True)
    w, rng = world()
    half = n_frames // 2
    xs = np.r_[np.arange(half) * 1.0 + 20, (half - 1 - np.arange(n_frames - half)) * 1.0 + 20.3]
    heading = np.r_[np.zeros(half), np.full(n_frames - half, np.pi)]
    vel = zipfile.ZipFile(os.path.join(out_dir, "data_odometry_velodyne.zip"), "w")
    poses = []
    for i, (x, h) in enumerate(zip(xs, heading)):
        p = w - [x, 0, 0]
        c, s = np.cos(-h), np.sin(-h)
        p = p @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
        p = p[np.linalg.norm(p[:, :2], axis=1) < 60]
        p = p[rng.random(len(p)) < 0.6]
        scan = np.c_[p, rng.uniform(0, 1, len(p))].astype(np.float32)
        vel.writestr(f"dataset/sequences/00/velodyne/{i:06d}.bin", scan.tobytes())
        poses.append([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, x])
    vel.close()
    with zipfile.ZipFile(os.path.join(out_dir, "data_odometry_poses.zip"), "w") as z:
        z.writestr("dataset/poses/00.txt", "\n".join(" ".join(map(str, r)) for r in poses))
    with zipfile.ZipFile(os.path.join(out_dir, "data_odometry_calib.zip"), "w") as z:
        z.writestr("dataset/sequences/00/calib.txt", "Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    print("fake KITTI written to", out_dir)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 240)
