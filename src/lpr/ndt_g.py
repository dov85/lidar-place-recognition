"""Spatially distributed sampling G (NDT-Transformer, ICRA 2021, Sec. III-B).

The official repository does not include this step. This implementation
follows the paper and a clarification from one of the authors
(D. Adolfsson, personal communication, Sept. 2026): "a binary search on the
resolution that results in some 0-10% more cells than required, then removing
redundant information".

Steps
  1. Binary search on the voxel size v so that the number of valid cells M
     falls in [k, (1 + over) * k]. Each occupied voxel gives one centroid; a
     cell is valid if at least `min_pts` original points lie within radius r
     of the centroid.
  2. For every valid centroid, the NDT cell (mean, covariance) is computed from
     all original points within radius r (paper: "resolution of 0.8 m").
  3. Redundant cells are removed until exactly k remain. The paper calls the
     criterion "highest mutual information" without defining it; here the
     redundancy of a cell is its largest Bhattacharyya coefficient (overlap of
     the two Gaussians) with one of its nearest cells, and the most redundant
     cell is removed greedily.

Assumptions that are NOT in the paper are marked with "ASSUMPTION".
"""
import numpy as np
from scipy.spatial import cKDTree

DEFAULTS = dict(
    k=2000,          # paper
    over=0.10,       # author clarification: 0-10% more cells before removal
    radius=0.8,      # ASSUMPTION: the paper's "resolution of 0.8 m" read as r
    min_pts=5,       # ASSUMPTION: a covariance needs a few points
    knn=8,           # ASSUMPTION: neighbours considered for redundancy
    cov_eps=1e-3,    # ASSUMPTION: m^2 added to the diagonal, redundancy only
)


def voxel_centroids(points, voxel):
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    cents = np.zeros((len(counts), 3))
    for d in range(3):
        cents[:, d] = np.bincount(inv, weights=points[:, d]) / counts
    return cents


def _valid_centroids(points, tree, voxel, radius, min_pts):
    cents = voxel_centroids(points, voxel)
    n_near = tree.query_ball_point(cents, radius, return_length=True)
    return cents[n_near >= min_pts]


def search_resolution(points, tree, k, over, radius, min_pts,
                      v_lo=0.05, v_hi=20.0, max_iter=40):
    """Binary search on the voxel size. Number of cells decreases with v."""
    lo_target, hi_target = k, int(np.floor((1.0 + over) * k))
    best = None  # smallest set with at least k cells seen so far
    for _ in range(max_iter):
        v = 0.5 * (v_lo + v_hi)
        cents = _valid_centroids(points, tree, v, radius, min_pts)
        m = len(cents)
        if m >= lo_target and (best is None or m < len(best[1])):
            best = (v, cents)
        if lo_target <= m <= hi_target:
            return v, cents
        if m > hi_target:
            v_lo = v          # too many cells -> larger voxels
        else:
            v_hi = v          # too few cells -> smaller voxels
    if best is not None:
        return best
    # Very sparse scan: even the finest voxels give fewer than k valid cells.
    return v_lo, _valid_centroids(points, tree, v_lo, radius, min_pts)


def cell_statistics(points, tree, centroids, radius):
    """Mean and covariance (ddof=1, as in Eq. 1 of the paper) around each centroid."""
    neigh = tree.query_ball_point(centroids, radius)
    means = np.empty((len(centroids), 3))
    covs = np.empty((len(centroids), 3, 3))
    for i, idx in enumerate(neigh):
        p = points[idx]
        mu = p.mean(axis=0)
        d = p - mu
        means[i] = mu
        covs[i] = d.T @ d / max(len(p) - 1, 1)
    return means, covs


def bhattacharyya_coefficient(mu1, c1, mu2, c2):
    """Overlap exp(-D_B) of pairs of Gaussians; all inputs batched on axis 0."""
    c = 0.5 * (c1 + c2)
    diff = (mu1 - mu2)[..., None]
    sol = np.linalg.solve(c, diff)
    term1 = 0.125 * (diff.transpose(0, 2, 1) @ sol).reshape(-1)
    _, ld = np.linalg.slogdet(c)
    _, ld1 = np.linalg.slogdet(c1)
    _, ld2 = np.linalg.slogdet(c2)
    term2 = 0.5 * (ld - 0.5 * (ld1 + ld2))
    return np.exp(-(term1 + term2))


def remove_redundant(means, covs, k, knn, cov_eps):
    """Greedily drop the cell with the largest overlap with a neighbour."""
    m = len(means)
    if m <= k:
        return np.arange(m)
    knn = min(knn, m - 1)
    _, nbr = cKDTree(means).query(means, knn + 1)
    nbr = nbr[:, 1:]                                   # drop self
    reg = covs + cov_eps * np.eye(3)
    i_idx = np.repeat(np.arange(m), knn)
    j_idx = nbr.reshape(-1)
    bc = bhattacharyya_coefficient(means[i_idx], reg[i_idx],
                                   means[j_idx], reg[j_idx]).reshape(m, knn)
    alive = np.ones(m, dtype=bool)
    for _ in range(m - k):
        red = np.where(alive[nbr], bc, 0.0).max(axis=1)
        red[~alive] = -1.0
        alive[np.argmax(red)] = False
    return np.flatnonzero(alive)


def ndt_cells(points, k=DEFAULTS["k"], over=DEFAULTS["over"], radius=DEFAULTS["radius"],
              min_pts=DEFAULTS["min_pts"], knn=DEFAULTS["knn"], cov_eps=DEFAULTS["cov_eps"],
              rng=None):
    """Run G on an (N, 3) cloud in metres. Returns means (k,3), covs (k,3,3), info."""
    tree = cKDTree(points)
    voxel, cents = search_resolution(points, tree, k, over, radius, min_pts)
    means, covs = cell_statistics(points, tree, cents, radius)
    n_before = len(means)
    keep = remove_redundant(means, covs, k, knn, cov_eps)
    means, covs = means[keep], covs[keep]
    padded = 0
    if len(means) < k:
        # ASSUMPTION: rare very sparse scans are padded by repeating random cells.
        rng = rng or np.random.default_rng(0)
        extra = rng.choice(len(means), k - len(means), replace=True)
        means, covs = np.concatenate([means, means[extra]]), np.concatenate([covs, covs[extra]])
        padded = len(extra)
    info = dict(voxel=float(voxel), cells_before_removal=int(n_before),
                removed=int(max(n_before - k, 0)), padded=int(padded), n_points=int(len(points)))
    return means, covs, info


def to_official_format(means, covs, scale=40.0):
    """(k, 12) float64 rows [mean(3), covariance(9)] as read by the official loader.

    ASSUMPTION: coordinates are divided by the crop half-size (40 m) so that the
    means lie in [-1, 1], like the normalised submaps the official code was
    written for; covariances are divided by scale**2 accordingly. The official
    loader (loading_pointclouds.load_pc_file) multiplies the covariance by 1000.
    """
    out = np.concatenate([means / scale, covs.reshape(len(covs), 9) / scale ** 2], axis=1)
    return out.astype(np.float64)


def from_official_format(arr, scale=40.0):
    arr = arr.reshape(-1, 12)
    return arr[:, :3] * scale, arr[:, 3:].reshape(-1, 3, 3) * scale ** 2
