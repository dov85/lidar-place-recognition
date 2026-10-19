"""NetVLAD heads, both with one interface: input (B, C, N, 1) local features, output (B, D).

NetVLAD      the head of BEVPlace++ (third_party/BEVPlace2/REIN.py): 64 clusters,
             descriptor 64 * C, intra-normalisation + L2; soft assignment by a 1x1 conv
             without bias; centroids and conv weights initialised from k-means
             (init_params, same code as the official one). The official forward loops over
             the clusters; here the same sum is computed as
               V_k = sum_i a_ik x_i - (sum_i a_ik) c_k
             (identical result, checked in tests/test_models.py).
NetVLADLoupe the head of NDT-Transformer (third_party/NDT-Transformer/models/NetVLAD.py):
             assignment x @ W -> BatchNorm -> softmax, separate centres, intra-normalisation
             + L2, then a fully connected layer to output_dim, BatchNorm and context
             gating. Random initialisation as in the official code. l2_out=True adds an L2
             normalisation of the output (not in the official code; used so that the
             triplet margin of the BEVPlace++ recipe means the same for both heads).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = ("vlad", "loupe")


class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=128):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.out_dim = num_clusters * dim

    def init_params(self, clsts, traindescs):
        """k-means centroids (K, C) and the descriptors they came from (official code)."""
        clsts_assign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clsts_assign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]
        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        self.conv.weight = nn.Parameter(
            torch.from_numpy(self.alpha * clsts_assign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        B, C = x.shape[:2]
        x_flat = x.reshape(B, C, -1)                                       # (B, C, N)
        a = F.softmax(self.conv(x).reshape(B, self.num_clusters, -1), dim=1)   # (B, K, N)
        vlad = torch.bmm(a, x_flat.transpose(1, 2))                        # (B, K, C)
        vlad = vlad - a.sum(-1).unsqueeze(-1) * self.centroids.unsqueeze(0)
        vlad = F.normalize(vlad, p=2, dim=2)        # intra-normalisation
        vlad = vlad.reshape(B, -1)
        return F.normalize(vlad, p=2, dim=1)


class GatingContext(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.bn1 = nn.BatchNorm1d(dim)

    def forward(self, x):
        return x * torch.sigmoid(self.bn1(x @ self.gating_weights))


class NetVLADLoupe(nn.Module):
    """Official NetVLADLoupe with add_batch_norm=True and gating=True (the settings of
    NDT-Transformer); parameter names as in the official code."""

    def __init__(self, feature_size=1024, cluster_size=64, output_dim=256, l2_out=False):
        super().__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.l2_out = l2_out
        s = 1 / math.sqrt(feature_size)
        self.cluster_weights = nn.Parameter(torch.randn(feature_size, cluster_size) * s)
        self.cluster_weights2 = nn.Parameter(torch.randn(1, feature_size, cluster_size) * s)
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * s)
        self.bn1 = nn.BatchNorm1d(cluster_size)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.context_gating = GatingContext(output_dim)
        self.out_dim = output_dim

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, self.feature_size, -1).transpose(1, 2)        # (B, N, D)
        N = x.shape[1]
        act = x @ self.cluster_weights                                  # (B, N, K)
        act = self.bn1(act.reshape(-1, self.cluster_size)).reshape(B, N, self.cluster_size)
        act = F.softmax(act, dim=-1)
        a = act.sum(-2, keepdim=True) * self.cluster_weights2           # (B, D, K)
        vlad = (act.transpose(2, 1) @ x).transpose(2, 1) - a            # (B, D, K)
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = F.normalize(vlad.reshape(B, -1), dim=1, p=2)
        vlad = self.bn2(vlad @ self.hidden1_weights)
        vlad = self.context_gating(vlad)
        return F.normalize(vlad, dim=1, p=2) if self.l2_out else vlad


def build_head(head, feat_dim, clusters=64, output_dim=256, l2_out=True):
    if head == "vlad":
        return NetVLAD(clusters, feat_dim)
    if head == "loupe":
        return NetVLADLoupe(feat_dim, clusters, output_dim, l2_out=l2_out)
    raise ValueError(f"head must be one of {HEADS}, not {head!r}")
