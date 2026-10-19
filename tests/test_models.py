"""Checks of src/lpr/models against the official code (CPU, random inputs).

  1. official_twin() with the weights of an official NDTNetVlad gives the same descriptors
     (with and without cov_transform);
  2. heads.NetVLAD (vectorised) gives the same descriptors as the NetVLAD of BEVPlace++
     (loop over the clusters), also after init_params;
  3. NDTNet with rotations=8: a scan rotated by 45 deg gives the same descriptor;
     every head / T-Net combination runs forward and backward.

  python tests/test_models.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "NDT-Transformer"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "BEVPlace2"))

from lpr.models import ndt_net  # noqa: E402
from lpr.models.heads import NetVLAD  # noqa: E402
from lpr.rotation import rotate_cells  # noqa: E402


def cells(b, n=2000, seed=0):
    """Plausible NDT input: means in [-1, 1], covariances SPD x 1000 (as the loader gives)."""
    rng = np.random.default_rng(seed)
    m = rng.uniform(-1, 1, (b, n, 3))
    m[..., 2] *= 0.05
    A = rng.normal(0, 0.01, (b, n, 3, 3))
    S = (A @ A.transpose(0, 1, 3, 2) + 1e-4 * np.eye(3)) * 1000
    return np.concatenate([m, S.reshape(b, n, 9)], -1).astype(np.float32)


def randomize_bn(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    for mod in model.modules():
        if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            mod.running_mean.copy_(torch.randn(mod.num_features, generator=g) * 0.1)
            mod.running_var.copy_(torch.rand(mod.num_features, generator=g) + 0.5)
            mod.weight.data.copy_(torch.rand(mod.num_features, generator=g) + 0.5)
            mod.bias.data.copy_(torch.randn(mod.num_features, generator=g) * 0.1)


def test_official_twin():
    import models.NDTNetVlad as official
    torch.manual_seed(0)
    off = official.NDTNetVlad(num_points=2000, output_dim=256, emb_dims=1024, layer_number=3)
    with torch.no_grad():   # a T-Net that is not the identity
        off.Transformer.point_trans.fc3.weight.normal_(0, 0.01)
        off.Transformer.point_trans.fc3.bias.normal_(0, 0.05)
    randomize_bn(off)
    off.eval()
    x = torch.from_numpy(cells(3))
    for cov in (False, True):
        off.Transformer.convariance_transform = cov
        mine = ndt_net.official_twin(cov_transform=cov)
        mine.load_state_dict(ndt_net.official_to_own(off.state_dict()), strict=True)
        mine.eval()
        with torch.no_grad():
            a = off(x.unsqueeze(1))
            b = mine(x)[2]
        err = (a - b).abs().max().item() / a.abs().max().item()
        print(f"official vs own (cov_transform={cov}): max relative difference {err:.2e}")
        assert err < 1e-4, err


def test_netvlad():
    from REIN import NetVLAD as OfficialVLAD
    torch.manual_seed(0)
    off, mine = OfficialVLAD(64, 128), NetVLAD(64, 128)
    mine.load_state_dict(off.state_dict())
    x = torch.nn.functional.normalize(torch.randn(2, 128, 30, 30), dim=1)
    with torch.no_grad():
        err = (off(x) - mine(x)).abs().max().item()
    rng = np.random.default_rng(0)
    descs = rng.normal(size=(500, 128)).astype(np.float32)
    clsts = descs[:64].copy()
    off.init_params(clsts, descs)
    mine.init_params(clsts, descs)
    with torch.no_grad():
        err2 = (off(x) - mine(x)).abs().max().item()
    print(f"NetVLAD official vs own: {err:.2e} (random), {err2:.2e} (after init_params)")
    assert err < 1e-5 and err2 < 1e-5


def test_rotations_and_backward():
    x = cells(2, n=400, seed=3)   # fewer cells: CPU memory (the network does not depend on N)
    xr = rotate_cells(x, 45.0).astype(np.float32)
    for head in ("vlad", "loupe"):
        for stn in (False, True):
            torch.manual_seed(0)
            net = ndt_net.NDTNet(stn=stn, rotations=8, head=head, feat_dim=128)
            randomize_bn(net)
            net.eval()
            with torch.no_grad():
                a = net(torch.from_numpy(x))[2]
                b = net(torch.from_numpy(xr))[2]
            err = (a - b).abs().max().item()
            net.train()
            loss = net(torch.from_numpy(cells(3, n=400, seed=5)))[2].pow(2).sum()
            loss.backward()
            print(f"head {head}, stn {stn}: descriptor {tuple(a.shape)}, scan rotated by 45 deg "
                  f"-> max difference {err:.2e}; backward ok")
            assert err < 1e-4, err   # the 8 copies of the rotated scan are the same 8 copies


if __name__ == "__main__":
    test_netvlad()
    test_official_twin()
    test_rotations_and_backward()
    print("all model checks passed")
