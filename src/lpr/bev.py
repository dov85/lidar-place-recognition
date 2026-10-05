"""BEV density images as in BEVPlace++ (third_party/BEVPlace2/datasets/gen_bev_images.py).

The official script loops over the points in Python; this is a vectorised version of the
same computation (checked pixel for pixel against the official one in tests/):
  1. crop |x|, |y|, |z| < 40 m (strict, as the official script);
  2. voxel down-sampling with 0.4 m voxels, Open3D semantics (voxel grid anchored at
     min_bound - voxel/2, each voxel replaced by the mean of its points);
  3. count the down-sampled points per 0.4 m pixel of a (2*40/0.4 + 1)^2 = 201 x 201
     grid, capped at 10; row = 100 - floor(x / 0.4), column = 100 - floor(y / 0.4);
  4. counts * 10, scaled so that the brightest pixel is 255 (float image, written
     with cv2.imwrite like the official script, which rounds to uint8).
"""
import numpy as np

BEV_PARAM_DOC = {
    "half_size": "crop half-size in x and y, m (official: 40)",
    "z_abs": "crop |z| below this, m (official: 40)",
    "voxel": "voxel down-sampling size, m (official: 0.4)",
    "res": "pixel size, m (official: 0.4)",
    "max_count": "cap on points per pixel (official: 10)",
    "ground": "ground removal before the image: none (official) | patchwork | ransac",
}


def crop(points, half_size=40.0, z_abs=40.0):
    """Keeps every column of the points (x, y, z[, intensity])."""
    p = points
    keep = (np.abs(p[:, 0]) < half_size) & (np.abs(p[:, 1]) < half_size) & (np.abs(p[:, 2]) < z_abs)
    return p[keep]


def voxel_down_sample(points, voxel):
    """Mean point of every occupied voxel, like open3d.geometry.PointCloud.voxel_down_sample."""
    if len(points) == 0:
        return points.astype(np.float64)
    p = points.astype(np.float64)
    vmin = p.min(axis=0) - voxel * 0.5
    idx = np.floor((p - vmin) / voxel).astype(np.int64)
    _, inv, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inv, p)
    return sums / counts[:, None]


def bev_image(points, half_size=40.0, res=0.4, max_count=10):
    """Float image (H x W) with values in [0, 255]; points are the down-sampled cloud."""
    max_ind = int(np.floor(half_size / res))
    n = 2 * max_ind + 1
    img = np.zeros((n, n), dtype=np.int64)
    if len(points):
        cols = max_ind - np.floor(points[:, 1] / res).astype(np.int64)
        rows = max_ind - np.floor(points[:, 0] / res).astype(np.int64)
        ok = (cols < n) & (rows < n) & (cols >= 0) & (rows >= 0)
        np.add.at(img, (rows[ok], cols[ok]), 1)
    img = np.minimum(img, max_count) * 10
    img = np.minimum(img, 255).astype(np.float64)
    m = img.max()
    return img / m * 255 if m > 0 else img


def make_bev(scan, half_size=40.0, z_abs=40.0, voxel=0.4, res=0.4, max_count=10, ground="none"):
    """scan: (N, >=3) raw KITTI points. Returns (float image, info dict)."""
    pts = crop(scan.astype(np.float32), half_size, z_abs)
    if ground != "none":
        from lpr.preprocess import remove_ground_patchwork, remove_ground_ransac
        pts = remove_ground_patchwork(pts) if ground == "patchwork" else remove_ground_ransac(pts)
    ds = voxel_down_sample(pts[:, :3], voxel)
    img = bev_image(ds, half_size, res, max_count)
    info = {"n_raw": int(len(scan)), "n_points": int(len(pts)), "n_voxels": int(len(ds)),
            "occupied_px": int((img > 0).sum())}
    return img, info
