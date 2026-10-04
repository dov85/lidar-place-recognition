"""KITTI Odometry helpers: scans, poses and the place-recognition split.

The split follows the BEVPlace++ evaluation code (kitti_dataset.py in the
official repository): frames [0, split) form the database, queries start
200 frames after the split point, and a match is correct within 5 m.
"""
import os
import numpy as np

# Split points used by BEVPlace++ (kitti_dataset.py, kitti_seq_split_points).
SPLIT_POINTS = {"00": 3000, "02": 3400, "05": 1000, "06": 600, "08": 1000}
QUERY_GAP = 200          # frames skipped after the split point
MATCH_THRESHOLD_M = 5.0  # a retrieved frame is correct if within this distance


def scan_path(kitti_root, seq, idx):
    return os.path.join(kitti_root, "sequences", seq, "velodyne", f"{idx:06d}.bin")


def load_scan(path):
    """Velodyne scan as (N, 4) float32: x, y, z, reflectance (sensor frame)."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_poses(kitti_root, seq):
    """Ground-truth poses as (N, 3, 4) matrices (left camera frame of frame 0)."""
    p = np.loadtxt(os.path.join(kitti_root, "poses", f"{seq}.txt"))
    return p.reshape(-1, 3, 4)


def positions(poses):
    """(N, 3) translation of each frame. Same entries (3, 7, 11) BEVPlace++ uses."""
    return poses[:, :, 3]


def num_frames(kitti_root, seq):
    d = os.path.join(kitti_root, "sequences", seq, "velodyne")
    return len([f for f in os.listdir(d) if f.endswith(".bin")])


def split_indices(seq, n_frames):
    """Database and query frame indices for one sequence."""
    split = SPLIT_POINTS[seq]
    db = np.arange(0, split)
    q = np.arange(split + QUERY_GAP, n_frames)
    return db, q
