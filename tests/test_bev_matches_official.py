"""Pixel-for-pixel check of lpr.bev against the official BEVPlace++ getBEV.

Needs open3d and opencv; the official script is loaded from third_party/BEVPlace2
(or from --official). Uses synthetic scans unless --scans points at KITTI .bin files.
  python tests/test_bev_matches_official.py [--official third_party/BEVPlace2] [--scans a.bin b.bin]
"""
import argparse
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr.bev import make_bev  # noqa: E402


def load_official(root):
    path = os.path.join(root, "datasets", "gen_bev_images.py")
    src = open(path).read().split('if __name__ == "__main__":')[0]
    mod = {}
    exec(compile(src, path, "exec"), mod)
    return mod["getBEV"]


def official_png(getBEV, scan, out):
    pcs = scan[:, :3]
    pcs = pcs[np.where(np.abs(pcs[:, 0]) < 40)[0], :]
    pcs = pcs[np.where(np.abs(pcs[:, 1]) < 40)[0], :]
    pcs = pcs[np.where((np.abs(pcs[:, 2]) < 40))[0], :]
    img, _, _ = getBEV(pcs.astype(np.float32))
    cv2.imwrite(out, img)
    return cv2.imread(out, 0)


def synthetic(seed):
    r = np.random.default_rng(seed)
    n = 120000
    ang = r.uniform(0, 2 * np.pi, n)
    dist = np.abs(r.normal(15, 12, n)) + 2
    z = r.normal(-1.5, 1.0, n)
    pts = np.stack([dist * np.cos(ang), dist * np.sin(ang), z, r.uniform(0, 1, n)], 1)
    walls = np.stack([r.uniform(-60, 60, 20000), np.full(20000, 7.3) + r.normal(0, .05, 20000),
                      r.uniform(-1.7, 3, 20000), r.uniform(0, 1, 20000)], 1)
    return np.vstack([pts, walls]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", default=os.path.join(os.path.dirname(__file__), "..",
                                                       "third_party", "BEVPlace2"))
    ap.add_argument("--scans", nargs="*", default=[])
    a = ap.parse_args()
    getBEV = load_official(a.official)
    scans = [np.fromfile(f, dtype=np.float32).reshape(-1, 4) for f in a.scans] or \
            [synthetic(s) for s in range(3)]
    tmp = tempfile.mkdtemp()
    worst = 0
    for i, scan in enumerate(scans):
        ref = official_png(getBEV, scan, os.path.join(tmp, f"ref{i}.png"))
        img, info = make_bev(scan)
        cv2.imwrite(os.path.join(tmp, f"ours{i}.png"), img)
        ours = cv2.imread(os.path.join(tmp, f"ours{i}.png"), 0)
        diff = np.abs(ref.astype(int) - ours.astype(int))
        worst = max(worst, diff.max())
        print(f"scan {i}: shape {ours.shape}, pixels differing {(diff > 0).sum()}, "
              f"max diff {diff.max()}, {info}")
    print("IDENTICAL" if worst == 0 else "DIFFERENT")
    sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
