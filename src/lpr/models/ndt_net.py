"""Own implementation of the NDT-Transformer network, with two additions.

Input: NDT cells of one scan, (B, N, 12) or (B, 1, N, 12) as the official loader gives them:
[mean (3, divided by half_size), covariance (9, x 1000)], N = 2000.

Layers (same as third_party/NDT-Transformer/models, same parameter shapes):
  T-Net (STN3d, optional)   3x3 transform of the means (and of the covariances,
                            T^T S T, with cov_transform)
  conv1 1x1  12 -> 256, BatchNorm, ReLU                       = x1
  TransformerEncoder        3 layers, d_model 256, 4 heads, feed-forward 1024, dropout 0.1
  conv2 1x1  [x1, transformer(x1)] 512 -> feat_dim, BatchNorm, ReLU
  head                      "loupe" (official NetVLADLoupe, 256-d) or "vlad" (the NetVLAD of
                            BEVPlace++, 64 * feat_dim), see heads.py

Additions:
  rotations = R > 1  REIN-like rotation equivariance, as REM in BEVPlace++: the cells are
                     rotated about z by 0, 360/R, ..., every copy goes through the same
                     encoder (T-Net if any, conv1, transformer, conv2), and the features of
                     each cell are the maximum over the R copies. Cells keep their order, so
                     no warp back is needed (BEVPlace++ rotates its feature map back). A scan
                     rotated by a multiple of 360/R gives exactly the same descriptor.
  local_norm         L2 normalisation of every cell feature before the head (BEVPlace++
                     normalises its feature map before NetVLAD); default on for "vlad".

The forward returns (local, local, global) like REIN in BEVPlace++, so the training code of
src/lpr/bevplace.py (cache, hard mining, k-means initialisation) works unchanged.
NDTNet(stn=True, rotations=1, head="loupe", feat_dim=1024, local_norm=False, l2_out=False)
is the official network: load_official() maps an official state_dict onto it
(checked in tests/test_models.py).
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import HEADS, build_head

NUM_CELLS = 2000


class STN3d(nn.Module):
    """Official STN3d with k=3, use_bn=False (the setting of NDT-Transformer)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, (1, 3))
        self.conv2 = nn.Conv2d(64, 128, (1, 1))
        self.conv3 = nn.Conv2d(128, 1024, (1, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()

    def forward(self, m):                      # m: (B, N, 3)
        x = m.unsqueeze(1)                     # (B, 1, N, 3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))              # (B, 1024, N, 1)
        x = x.amax(dim=(2, 3))                 # max over the cells
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) + torch.eye(3, device=x.device, dtype=x.dtype).reshape(1, 9)
        return x.reshape(-1, 3, 3)


class Encoder(nn.Module):
    def __init__(self, feat_dim=1024, stn=True, cov_transform=False, layers=3):
        super().__init__()
        self.stn = STN3d() if stn else None
        self.cov_transform = cov_transform
        self.conv1 = nn.Conv2d(12, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024,
                                           activation="relu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers,
                                                 enable_nested_tensor=False)
        self.conv2 = nn.Conv2d(512, feat_dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(feat_dim)

    def forward(self, x):                      # (B, N, 12) -> (B, feat_dim, N, 1)
        m, s = x[..., :3], x[..., 3:]
        if self.stn is not None:
            T = self.stn(m)
            m = m @ T
            if self.cov_transform:
                B, N = s.shape[:2]
                S = s.reshape(B, N, 3, 3)
                s = (T.transpose(1, 2).unsqueeze(1) @ S @ T.unsqueeze(1)).reshape(B, N, 9)
        h = torch.cat([m, s], dim=-1).transpose(1, 2).unsqueeze(-1)    # (B, 12, N, 1)
        x1 = F.relu(self.bn1(self.conv1(h)))                           # (B, 256, N, 1)
        t = self.transformer(x1.squeeze(-1).transpose(1, 2))          # (B, N, 256)
        t = t.transpose(1, 2).unsqueeze(-1)
        return F.relu(self.bn2(self.conv2(torch.cat([x1, t], dim=1))))


def rotate_cells_torch(x, angle):
    """x (B, N, 12); rotation about z by angle (radians): m -> R m, S -> R S R^T."""
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=x.device,
                     dtype=x.dtype)
    B, N = x.shape[:2]
    m = x[..., :3] @ R.T
    S = R @ x[..., 3:].reshape(B, N, 3, 3) @ R.T
    return torch.cat([m, S.reshape(B, N, 9)], dim=-1)


class NDTNet(nn.Module):
    def __init__(self, stn=False, rotations=8, head="vlad", feat_dim=128, cov_transform=False,
                 local_norm=None, l2_out=True, clusters=64, output_dim=256):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, not {head!r}")
        if rotations < 1:
            raise ValueError("rotations must be >= 1")
        self.rotations = int(rotations)
        self.angles = [2 * math.pi * k / self.rotations for k in range(self.rotations)]
        self.local_norm = (head == "vlad") if local_norm is None else bool(local_norm)
        self.encoder = Encoder(feat_dim, stn=stn, cov_transform=cov_transform)
        self.pooling = build_head(head, feat_dim, clusters, output_dim, l2_out=l2_out)
        self.local_feat_dim = feat_dim
        self.global_feat_dim = self.pooling.out_dim

    def local_features(self, x):
        if x.dim() == 4:                       # (B, 1, N, 12) of the official code
            x = x.squeeze(1)
        B = x.shape[0]
        if self.rotations == 1:
            f = self.encoder(x)
        else:
            xs = torch.cat([rotate_cells_torch(x, a) for a in self.angles], dim=0)
            f = self.encoder(xs)               # (R*B, C, N, 1)
            f = f.reshape(self.rotations, B, *f.shape[1:]).amax(dim=0)
        return F.normalize(f, dim=1) if self.local_norm else f

    def forward(self, x):
        local = self.local_features(x)
        return local, local, self.pooling(local)


# ---------------------------------------------------------------- model from a run / official
MODEL_KEYS = ("stn", "rotations", "head", "feat_dim", "cov_transform")


def build(params):
    """NDTNet from the training parameters of a run (keys MODEL_KEYS)."""
    return NDTNet(**{k: params[k] for k in MODEL_KEYS if k in params})


def run_params(ckpt_path):
    """Training parameters of the run a checkpoint belongs to, if it is a run of this model
    (config.json next to the checkpoint with method "myndt"); None otherwise."""
    cfg = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "config.json")
    try:
        c = json.load(open(cfg))
    except (OSError, ValueError):
        return None
    return c.get("params") if c.get("method") == "myndt" else None


def official_twin(cov_transform=False):
    """The official NDT-Transformer architecture in this implementation."""
    return NDTNet(stn=True, rotations=1, head="loupe", feat_dim=1024,
                  cov_transform=cov_transform, local_norm=False, l2_out=False)


def official_to_own(state_dict):
    """Keys of an official NDTNetVlad state_dict -> keys of official_twin()."""
    out = {}
    for k, v in state_dict.items():
        k = (k.replace("Transformer.point_trans.", "encoder.stn.")
              .replace("Transformer.transformer_encoder.", "encoder.transformer.")
              .replace("Transformer.", "encoder.")
              .replace("net_vlad.", "pooling."))
        if k.startswith("pooling.cluster_biases") or k.startswith("encoder.stn.bn"):
            continue
        out[k] = v
    return out
