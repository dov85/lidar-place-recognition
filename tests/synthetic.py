"""Synthetic street-like LiDAR scans, only for testing the code without KITTI."""
import numpy as np


def _range_density(pts, rng, keep_at_10m=1.0):
    r = np.linalg.norm(pts[:, :2], axis=1) + 1e-6
    p = np.clip((10.0 / r) ** 2 * keep_at_10m, 0, 1)
    return pts[rng.random(len(pts)) < p]


def synthetic_scan(seed=0, n=250_000, sensor_height=1.73):
    rng = np.random.default_rng(seed)
    parts = []
    # ground
    g = np.c_[rng.uniform(-60, 60, n // 2), rng.uniform(-60, 60, n // 2),
              np.full(n // 2, -sensor_height) + rng.normal(0, 0.02, n // 2)]
    parts.append(g)
    # building facades on both sides of the street, with gaps
    for side in (-1, 1):
        y0 = side * rng.uniform(7, 12)
        x = rng.uniform(-60, 60, n // 5)
        x = x[(np.sin(x / 7.0 + seed) > -0.6)]
        parts.append(np.c_[x, np.full(len(x), y0) + rng.normal(0, 0.03, len(x)),
                           rng.uniform(-sensor_height, 8, len(x))])
    # parked cars (boxes)
    for _ in range(15):
        cx, cy = rng.uniform(-40, 40), rng.choice([-1, 1]) * rng.uniform(3, 5)
        m = 2000
        b = np.c_[rng.uniform(-2, 2, m), rng.uniform(-0.9, 0.9, m), rng.uniform(0, 1.5, m)]
        face = rng.integers(0, 3, m)
        b[face == 0, 0] = np.sign(b[face == 0, 0]) * 2
        b[face == 1, 1] = np.sign(b[face == 1, 1]) * 0.9
        b[face == 2, 2] = 1.5
        parts.append(b + [cx, cy, -sensor_height])
    # poles
    for _ in range(25):
        cx, cy = rng.uniform(-40, 40), rng.uniform(-10, 10)
        m = 400
        a = rng.uniform(0, 2 * np.pi, m)
        parts.append(np.c_[cx + 0.15 * np.cos(a), cy + 0.15 * np.sin(a),
                           rng.uniform(-sensor_height, 4, m)])
    pts = np.concatenate(parts)
    pts = _range_density(pts, rng)
    refl = rng.uniform(0, 1, len(pts))
    return np.c_[pts, refl].astype(np.float32)
