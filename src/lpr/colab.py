"""Helpers shared by the three notebooks (Colab): run a script, put a variant on this machine.

Data path of every variant:
  Azure blob container  --scripts/fetch_from_azure.py-->  Drive <drive>/<method>/<variant>/*.tar
                        --tar -x-->  this machine <local_root>/<method>/<variant>/<seq>/
or, without access to the original Drive and Azure, from the public link of the project folder
(PUBLIC_URL in the notebooks, src/lpr/public_drive.py):
  public Drive link  --public_drive.fetch-->  local mirror <drive>/<method>/<variant>/*.tar  --> ...
"""
import os
import subprocess
import sys

from . import experiments as ex
from . import public_drive as pdrive

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXT = {"ndt": "000000.bin", "bev": "000000.png"}
PUBLIC_URL = ""     # set by the notebooks: the shared project folder (read-only), "" = not used


def run(args, tail=40):
    """Run `python <args>` in the code folder; print the last lines; return the return code."""
    res = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    out = res.stdout if res.returncode == 0 else res.stdout + res.stderr
    print("\n".join(out.strip().splitlines()[-tail:]))
    return res.returncode


def use_public(url, drive):
    """Work from the public link of the project folder: `drive` becomes a local mirror of it.
    The configurations come now (so the existing variants and runs are found by their
    parameters); the cells, images and checkpoints come when they are needed."""
    global PUBLIC_URL
    PUBLIC_URL = url
    if url:
        os.makedirs(drive, exist_ok=True)
        pdrive.mirror_configs(url, drive)


def from_public(drive, relpath):
    """One file of the project folder into the mirror (if PUBLIC_URL is set); its path or None."""
    if not PUBLIC_URL:
        return None
    return pdrive.fetch(PUBLIC_URL, relpath, drive, quiet=False)


def checkpoint(drive, method, run, ckpt="last"):
    """The checkpoint of a run on this machine: from the public link if it is not there yet.
    Returns `ckpt` unchanged when it is already available (or there is no public link)."""
    if PUBLIC_URL and not (os.path.isabs(ckpt) and os.path.exists(ckpt)):
        path = pdrive.fetch_checkpoint(PUBLIC_URL, drive, method, run, ckpt)
        if path:
            return os.path.basename(path)
    return ckpt


def has_checkpoint(run_dir):
    """A run with at least one checkpoint, here or (with the public link) in the shared folder."""
    return bool(pdrive.checkpoints(run_dir))


def to_machine(drive, method, variant, seqs, sas=None, local_root="/content"):
    """The sequences `seqs` of a variant on this machine (<local_root>/<method>/<variant>/).
    What is not on Drive yet comes from the public link (PUBLIC_URL) or, with `sas`, from the
    blob container. Returns the folder and the sequences that are there."""
    vdir = os.path.join(drive, method, variant)
    local = os.path.join(local_root, method, variant)
    tar = lambda s: os.path.join(vdir, ex.archive_name(method, s))  # noqa: E731
    from_public(drive, f"{method}/{variant}/config.json")
    for s in seqs:
        if not os.path.exists(tar(s)):
            from_public(drive, f"{method}/{variant}/{ex.archive_name(method, s)}")
    missing = [s for s in seqs if not os.path.exists(tar(s))]
    if missing and sas:
        run(["scripts/fetch_from_azure.py", "--sas_url", sas, "--method", method,
             "--variant", variant, "--seqs", *missing, "--drive", drive])
    os.makedirs(local, exist_ok=True)
    for s in seqs:
        if os.path.exists(tar(s)) and not os.path.exists(os.path.join(local, s, EXT[method])):
            subprocess.run(["tar", "-xf", tar(s), "-C", local], check=True)
    if os.path.exists(os.path.join(vdir, "config.json")):
        subprocess.run(["cp", os.path.join(vdir, "config.json"), os.path.join(local, "config.json")], check=True)
    have = [s for s in seqs if os.path.exists(os.path.join(local, s, EXT[method]))]
    print(f"{method}/{variant}: on this machine for {have}"
          + ("" if len(have) == len(seqs) else f", missing {sorted(set(seqs) - set(have))}"))
    return local, have


def sparse_to_machine(drive, method, variant, beams, seqs, sas=None, local_root="/content"):
    """The 32 / 16-beam variants of a variant, next to it on this machine (scripts/fetch_sparse.py)."""
    if beams:
        run(["scripts/fetch_sparse.py", "--method", method, "--variant", variant, "--drive", drive,
             "--data_root", os.path.join(local_root, method), "--beams", *beams, "--seqs", *seqs]
            + (["--sas_url", sas] if sas else []) + (["--public_url", PUBLIC_URL] if PUBLIC_URL else []))
