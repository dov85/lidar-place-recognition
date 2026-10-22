"""Fewer laser beams: a 64-beam KITTI scan (Velodyne HDL-64E) reduced to 32 or 16 beams.

KITTI .bin files carry no ring (laser) index, so it is recovered:
  1. from the order of the points: a scan is stored ring after ring, each ring one sweep in
     azimuth, so a new ring starts where the azimuth wraps around (a jump of more than pi
     against the direction of the sweep, after at least MIN_RING_POINTS points);
  2. if that does not give a plausible number of rings (EXPECTED_RINGS +- 8), from the
     elevation angle atan2(z, sqrt(x^2 + y^2)) of each point, assigned to the nearest of the
     64 nominal HDL-64E elevation angles.
The rings are then ranked by their median elevation (top to bottom) and every
(rings / beams)-th one is kept: 32 beams = every 2nd ring, 16 = every 4th, i.e. a sensor
with the same field of view and twice / four times the vertical spacing.
"""
import numpy as np

EXPECTED_RINGS = 64
MIN_RING_POINTS = 200

# nominal HDL-64E elevation angles (deg): upper block +2 .. -8.33 in 1/3 deg steps,
# lower block -8.83 .. -24.33 in 1/2 deg steps
HDL64_ELEVATION_DEG = np.r_[2.0 - np.arange(32) / 3.0, -8.83 - np.arange(32) * 0.5]


def ring_from_order(points):
    """Ring index per point from the azimuth wrap-arounds of the scan order."""
    yaw = np.arctan2(points[:, 1], points[:, 0])
    d = np.diff(yaw)
    direction = np.sign(np.median(d)) or 1.0       # the sweep turns this way
    cand = np.flatnonzero(d * direction < -np.pi) + 1   # wrap: big jump against the sweep
    starts, last = [], 0
    for c in cand:
        if c - last >= MIN_RING_POINTS:
            starts.append(c)
            last = c
    ring = np.zeros(len(points), dtype=np.int64)
    if starts:
        ring[np.asarray(starts)] = 1
        ring = np.cumsum(ring)
    return ring


def ring_from_elevation(points):
    elev = np.degrees(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))
    return np.abs(elev[:, None] - HDL64_ELEVATION_DEG[None, :]).argmin(axis=1)


def ring_index(points):
    """(ring index per point, number of rings, method used)."""
    ring = ring_from_order(points)
    n = int(ring.max()) + 1 if len(ring) else 0
    if abs(n - EXPECTED_RINGS) <= 8:
        return ring, n, "order"
    ring = ring_from_elevation(points)
    return ring, EXPECTED_RINGS, "elevation"


def drop_beams(points, beams, full=EXPECTED_RINGS):
    """Keep `beams` of the scan's rings (every full/beams-th by elevation). Returns
    (points kept, info)."""
    if beams >= full:
        return points, {"rings_found": None, "ring_method": None}
    if full % beams:
        raise ValueError(f"beams must divide {full}, not {beams}")
    ring, n, method = ring_index(points)
    elev = np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1]))
    ids = np.unique(ring)
    med = np.array([np.median(elev[ring == r]) for r in ids])
    rank = np.empty(len(ids), dtype=np.int64)
    rank[np.argsort(-med)] = np.arange(len(ids))         # 0 = highest ring
    keep_rings = ids[rank % (full // beams) == 0]
    keep = np.isin(ring, keep_rings)
    return points[keep], {"rings_found": n, "ring_method": method,
                          "rings_kept": int(len(keep_rings))}
