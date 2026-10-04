"""Point-cloud preprocessing applied before NDT cell generation.

Two steps, both documented as design choices of this project (the
NDT-Transformer paper does not specify them for single KITTI scans):
  1. crop to the same 80 m x 80 m square that BEVPlace++ projects to its BEV image;
  2. ground removal with Patchwork++ (RANSAC plane fit as a fallback).
"""
import numpy as np

CROP_HALF_SIZE_M = 40.0
KITTI_SENSOR_HEIGHT_M = 1.73

_patchwork = None


def crop_box(points, half_size=CROP_HALF_SIZE_M):
    m = (np.abs(points[:, 0]) <= half_size) & (np.abs(points[:, 1]) <= half_size)
    return points[m]


def _get_patchwork(sensor_height):
    global _patchwork
    if _patchwork is None:
        import pypatchworkpp
        params = pypatchworkpp.Parameters()
        params.verbose = False
        params.sensor_height = sensor_height
        _patchwork = pypatchworkpp.patchworkpp(params)
    return _patchwork


def remove_ground_patchwork(points4, sensor_height=KITTI_SENSOR_HEIGHT_M):
    pw = _get_patchwork(sensor_height)
    pw.estimateGround(np.ascontiguousarray(points4, dtype=np.float64))
    idx = np.asarray(pw.getNongroundIndices(), dtype=np.int64)
    return points4[idx]


def remove_ground_ransac(points4, sensor_height=KITTI_SENSOR_HEIGHT_M,
                         dist_thr=0.2, iters=100, seed=0):
    """Fit one plane to the low points and drop everything close to it."""
    rng = np.random.default_rng(seed)
    xyz = points4[:, :3]
    cand = xyz[xyz[:, 2] < -sensor_height + 0.7]
    if len(cand) < 3:
        return points4
    best_n, best_d, best_cnt = None, None, -1
    for _ in range(iters):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-6 or abs(n[2]) / norm < 0.9:   # keep near-horizontal planes only
            continue
        n = n / norm
        d = -n @ p[0]
        cnt = np.count_nonzero(np.abs(cand @ n + d) < dist_thr)
        if cnt > best_cnt:
            best_n, best_d, best_cnt = n, d, cnt
    if best_n is None:
        return points4
    return points4[np.abs(xyz @ best_n + best_d) >= dist_thr]


def preprocess(points4, ground="patchwork", half_size=CROP_HALF_SIZE_M):
    """Crop and remove the ground. Returns (M, 3) float64 xyz."""
    pts = crop_box(points4, half_size)
    if ground == "patchwork":
        pts = remove_ground_patchwork(pts)
    elif ground == "ransac":
        pts = remove_ground_ransac(pts)
    elif ground != "none":
        raise ValueError(f"unknown ground removal method: {ground}")
    return pts[:, :3].astype(np.float64)
