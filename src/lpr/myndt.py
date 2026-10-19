"""Data of the own NDT network (src/lpr/models/ndt_net.py) for the BEVPlace++ recipe.

TrainingDataset is the BEVPlace++ training set of src/lpr/bevplace.py (positives < 5 m,
negatives > 7 m among the training frames, hard mining from the descriptor cache), with NDT
cells instead of BEV images:
  - cells read as the official NDT-Transformer loader reads them (load_cells);
  - rot_aug: every training cloud rotated about z by its own random angle (rotate_cells);
  - rot_points = N: every training cloud taken from a random one of the N point-rotated copies
    of its scan (<ndt_dir>/<seq>_rot<N>/<idx>_<c>.bin), as in train_ndt_kitti.py.
As in BEVPlace++, the descriptor cache for hard mining and the k-means initialisation see
the training samples as the training does (rotated, from the copies); evaluation uses the
original cells (InferDataset).
"""
import os

import numpy as np
import scipy.ndimage
import torch.utils.data as data

from lpr import bevplace as bp
from lpr.experiments import rot_dir_name
from lpr.rotation import rotate_cells

NUM_CELLS = 2000


def load_cells(path, number_points=NUM_CELLS):
    """Official load_pc_file: float64 rows [mean(3), covariance(9)], resampled to 2000 rows
    if needed, covariances x 1000."""
    pc = np.fromfile(path, dtype=np.float64).reshape(-1, 12)
    if pc.shape[0] != number_points:
        zoom = number_points / float(pc.shape[0])
        pc = scipy.ndimage.zoom(pc, [zoom, 1], order=3, mode="constant", cval=0.0, prefilter=True)
    pc[:, 3:12] = pc[:, 3:12] * 1000
    return pc


def cell_paths(ndt_dir, seq):
    d = os.path.join(ndt_dir, seq)
    n = len([f for f in os.listdir(d) if f.endswith(".bin")])
    return [os.path.join(d, f"{i:06d}.bin") for i in range(n)]


class TrainingDataset(bp.TrainingDataset):
    def __init__(self, ndt_dir, seq, poses, train_frames=3000, pos_thr=5.0, neg_thr=7.0,
                 num_neg=10, rot_aug=True, rot_points=0):
        super().__init__(cell_paths(ndt_dir, seq), poses, train_frames, pos_thr, neg_thr,
                         num_neg, rot_aug)
        self.ndt_dir, self.seq, self.rot_points = ndt_dir, seq, int(rot_points)
        if self.rot_points:
            rd = os.path.join(ndt_dir, rot_dir_name(seq, self.rot_points))
            missing = [i for i in range(len(self.imgs_path)) for c in range(self.rot_points)
                       if not os.path.exists(os.path.join(rd, f"{i:06d}_{c}.bin"))]
            if missing:
                raise SystemExit(f"{len(missing)} point-rotated copies missing in {rd} "
                                 f"(e.g. frame {missing[0]}): build them first")

    def load(self, i):
        path = self.imgs_path[i]
        if self.rot_points:
            path = os.path.join(self.ndt_dir, rot_dir_name(self.seq, self.rot_points),
                                f"{i:06d}_{np.random.randint(self.rot_points)}.bin")
        x = load_cells(path)
        if self.rot_aug:
            x = rotate_cells(x, np.random.uniform(0, 360))
        return x.astype(np.float32)


class InferDataset(data.Dataset):
    """Cells of every scan of a sequence (evaluation), unrotated."""

    def __init__(self, paths, angles=None):
        self.paths = list(paths)
        self.angles = angles

    def __getitem__(self, index):
        x = load_cells(self.paths[index])
        if self.angles is not None:
            x = rotate_cells(x, self.angles[index])
        return x.astype(np.float32), index

    def __len__(self):
        return len(self.paths)
