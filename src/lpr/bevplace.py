"""BEVPlace++ training pieces, following the official code (third_party/BEVPlace2).

Taken from the official repository (main.py, kitti_dataset.py), unchanged in substance:
  - TrainingDataset: positives < 5 m, negatives > 7 m among the training frames,
    10 negatives per query; random 0-360 deg rotation of every image; hard mining
    (hardest 10 negatives by descriptor distance, a random positive) once the epoch's
    descriptor cache exists;
  - collate_fn, TripletLoss (margin 0.3, max over the negatives of each query),
  - train_epoch: descriptor cache of all training frames at the start of every epoch,
    then one pass over the frames in random order, loss / batch size, Adam;
  - get_clusters: NetVLAD initialisation by k-means (64 clusters) on 10000 local
    features from 400 training images.
The network itself (REIN) is imported from the official repository.

Differences, all bookkeeping: paths are given explicitly; the descriptor cache is kept
in memory instead of an HDF5 file; batches come from a saved permutation so a training
interrupted inside an epoch can resume (the cache is then rebuilt with the current
model); no NCLT validation and no "best checkpoint" chosen on the test sequences.
"""
import os
import sys
from math import ceil

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader


def official_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party",
                        "BEVPlace2")


def build_model(official=None):
    sys.path.insert(0, official or official_dir())
    from REIN import REIN  # noqa: E402  (official network)
    return REIN()


def image_paths(bev_dir, seq):
    d = os.path.join(bev_dir, seq)
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png") and not f.startswith(".")]


def _rotate(img):
    mat = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), np.random.randint(0, 360), 1)
    return cv2.warpAffine(img, mat, img.shape[:2])


class TrainingDataset(data.Dataset):
    def __init__(self, img_paths, poses, train_frames=3000, pos_thr=5.0, neg_thr=7.0,
                 num_neg=10, rot_aug=True):
        super().__init__()
        self.imgs_path = list(img_paths)[:train_frames]
        self.poses = np.asarray(poses).reshape(len(poses), -1)[:train_frames]
        self.num_neg = num_neg
        self.rot_aug = rot_aug
        self.positives, self.negatives = [], []
        for qi in range(len(self.poses)):
            dises = np.sqrt(np.sum(((self.poses[qi] - self.poses) ** 2)[:, [3, 7, 11]], axis=1))
            indexes = np.argsort(dises)
            remap_index = indexes[np.where(dises[indexes] < pos_thr)[0]]
            self.positives.append(remap_index[1:])  # exclude the query itself
            self.negatives.append(indexes[np.where(dises[indexes] > neg_thr)[0]])
        self.mining = False
        self.h5feat = None  # descriptor cache (official: HDF5 file read into memory)

    def __getitem__(self, index):
        if self.mining:
            q_feat = self.h5feat[index]
            pos_idx = np.random.choice(self.positives[index], 1)[0]
            neg_feat = self.h5feat[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1, -1) - neg_feat) ** 2, axis=1))
            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg - 1:-1]
            neg_idx = self.negatives[index][dis_inc_index_tmp[:self.num_neg]]
        else:
            pos_idx = self.positives[index][0]
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]

        def load(i):
            img = cv2.imread(self.imgs_path[i])
            if self.rot_aug:
                img = _rotate(img)
            return img.transpose(2, 0, 1)

        query = load(index).astype(np.float32) / 256
        positive = load(pos_idx).astype(np.float32) / 256
        negatives = torch.stack([torch.from_numpy((load(i) / 256).astype(np.float32))
                                 for i in neg_idx], 0)
        return query, positive, negatives, index

    def __len__(self):
        return len(self.poses)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    query, positive, negatives, indices = zip(*batch)
    query = data.dataloader.default_collate(np.array(query))
    positive = data.dataloader.default_collate(np.array(positive))
    return query, positive, torch.cat(negatives, 0), list(indices)


class InferDataset(data.Dataset):
    """Evaluation images: grey, / 256, repeated to 3 channels (official InferDataset)."""

    def __init__(self, img_paths):
        self.imgs_path = list(img_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index], 0).astype(np.float32) / 256
        return img[np.newaxis].repeat(3, 0), index

    def __len__(self):
        return len(self.imgs_path)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        return F.relu(pos_dist - neg_dist + self.margin)


@torch.no_grad()
def infer(model, dataset, device, batch=32, workers=2):
    model.eval()
    out = []
    for imgs, _ in DataLoader(dataset, num_workers=workers, batch_size=batch, shuffle=False):
        _, _, g = model(imgs.to(device))
        out.append(g.cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def build_cache(model, train_set, device, batch=4, workers=2):
    """Descriptors of every training frame (official: start of every epoch)."""
    model.eval()
    train_set.mining = False
    feats = np.zeros((len(train_set), model.global_feat_dim), dtype=np.float32)
    loader = DataLoader(train_set, num_workers=workers, batch_size=batch, shuffle=False,
                        collate_fn=collate_fn)
    for query, _, _, indices in loader:
        _, _, g = model(query.to(device))
        feats[indices, :] = g.cpu().numpy()
    train_set.h5feat = feats
    train_set.mining = True


def train_batches(model, train_set, batches, optimizer, device, batch_size, margin=0.3,
                  workers=2, log=None, on_iter=None, start=0, split=None):
    """One pass over the given index batches (official train_epoch, after the cache).

    split: [False] holds the "one forward pass per batch" flag. If a batch of 4 x 12 images
    does not fit in GPU memory, it is set to True and every tuple (12 images) gets its own
    forward/backward pass, with the gradients accumulated: the same loss and update, except
    that BatchNorm sees 12 images at a time instead of 48.
    """
    split = split if split is not None else [False]
    criterion = TripletLoss(margin).to(device)
    loader = DataLoader(train_set, num_workers=workers, batch_sampler=batches[start:],
                        collate_fn=collate_fn)
    model.train()
    losses = []

    def tuple_loss(gq, gp, gn, num_negs):
        loss = 0
        for i in range(len(gq)):
            loss += torch.max(criterion(gq[i], gp[i], gn[num_negs * i:num_negs * (i + 1)]))
        return loss / batch_size

    for it, (query, positives, negatives, _) in enumerate(loader, start + 1):
        B = query.shape[0]
        num_negs = negatives.shape[0] // B
        optimizer.zero_grad()
        total = None
        if not split[0]:
            try:
                inp = torch.cat([query, positives, negatives]).to(device)
                _, _, g = model(inp)
                gq, gp, gn = torch.split(g, [B, B, negatives.shape[0]])
                loss = tuple_loss(gq, gp, gn, num_negs)
                loss.backward()
                total = loss.item()
            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory with the full batch -> one tuple per forward pass",
                      flush=True)
                del inp
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                split[0] = True
        if split[0]:
            total = 0.0
            for i in range(B):
                negs = negatives[num_negs * i:num_negs * (i + 1)]
                inp = torch.cat([query[i:i + 1], positives[i:i + 1], negs]).to(device)
                _, _, g = model(inp)
                gq, gp, gn = torch.split(g, [1, 1, num_negs])
                loss = tuple_loss(gq, gp, gn, num_negs)
                loss.backward()
                total += loss.item()
        optimizer.step()
        losses.append(total)
        if log:
            log(it, losses)
        if on_iter:
            on_iter(it, losses)
    optimizer.zero_grad()
    return losses


@torch.no_grad()
def get_clusters(model, cluster_set, device, batch=128, workers=2, n_descriptors=10000,
                 n_per_image=25, niter=100, seed=None):
    """NetVLAD centroids (official getClusters); returns (centroids, descriptors)."""
    import faiss
    n_im = ceil(n_descriptors / n_per_image)
    sampler = data.SubsetRandomSampler(np.random.choice(len(cluster_set), n_im, replace=False))
    loader = DataLoader(cluster_set, num_workers=workers, batch_size=batch, shuffle=False,
                        sampler=sampler, collate_fn=collate_fn)
    model.eval()
    cluster_set.mining = False
    all_feats = np.zeros((n_descriptors, 128), dtype=np.float32)
    for iteration, (query, _, _, _) in enumerate(loader, 1):
        local_feat, _, _ = model(query.to(device))
        local_feat = local_feat.view(query.size(0), 128, -1).permute(0, 2, 1)
        batchix = (iteration - 1) * batch * n_per_image
        for ix in range(local_feat.size(0)):
            sample = np.random.choice(local_feat.size(1), n_per_image, replace=False)
            startix = batchix + ix * n_per_image
            all_feats[startix:startix + n_per_image, :] = local_feat[ix, sample, :].cpu().numpy()
    kw = {} if seed is None else {"seed": int(seed)}
    kmeans = faiss.Kmeans(128, 64, niter=niter, verbose=False, **kw)
    kmeans.train(all_feats)
    return kmeans.centroids, all_feats
