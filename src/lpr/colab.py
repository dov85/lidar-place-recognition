"""Helpers shared by the three notebooks (Colab): run a script, put a variant on this machine.

Data path of every variant:
  Azure blob container  --scripts/fetch_from_azure.py-->  Drive <drive>/<method>/<variant>/*.tar
                        --tar -x-->  this machine <local_root>/<method>/<variant>/<seq>/
"""
import os
import subprocess
import sys

from . import experiments as ex

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXT = {"ndt": "000000.bin", "bev": "000000.png"}


def run(args, tail=40):
    """Run `python <args>` in the code folder; print the last lines; return the return code."""
    res = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    out = res.stdout if res.returncode == 0 else res.stdout + res.stderr
    print("\n".join(out.strip().splitlines()[-tail:]))
    return res.returncode


def to_machine(drive, method, variant, seqs, sas=None, local_root="/content"):
    """The sequences `seqs` of a variant on this machine (<local_root>/<method>/<variant>/).
    What is not on Drive yet is copied from the blob container first (needs `sas`).
    Returns the folder and the sequences that are there."""
    vdir = os.path.join(drive, method, variant)
    local = os.path.join(local_root, method, variant)
    tar = lambda s: os.path.join(vdir, ex.archive_name(method, s))  # noqa: E731
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
            + (["--sas_url", sas] if sas else []))
