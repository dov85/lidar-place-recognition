"""Rotations about the vertical axis of NDT cells and BEV images (augmentation, compass)."""
import numpy as np


def rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def rotate_cells(x, deg):
    """x: (k, 12) or (n, k, 12) NDT rows [mean(3), covariance(9)] in the Velodyne frame
    (z up); deg: one angle, or one per cloud. Means -> R m, covariances -> R S R^T."""
    x = np.asarray(x)
    single = x.ndim == 2
    xs = x[None] if single else x
    degs = np.broadcast_to(np.asarray(deg, dtype=float), (len(xs),))
    out = np.empty_like(xs)
    for i, (c, d) in enumerate(zip(xs, degs)):
        R = rz(d)
        S = c[:, 3:].reshape(-1, 3, 3)
        out[i, :, :3] = c[:, :3] @ R.T
        out[i, :, 3:] = (R @ S @ R.T).reshape(-1, 9)
    return out[0] if single else out


def rotate_bev(img, deg):
    """BEV image of lpr.bev / BEVPlace++ (row = -x, column = -y, x up): the image of the
    points rotated by +deg about z is this image rotated by +deg (counter-clockwise)."""
    import cv2   # only here: the NDT build machines have no OpenCV
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), float(deg), 1)
    return cv2.warpAffine(img, mat, (w, h))
