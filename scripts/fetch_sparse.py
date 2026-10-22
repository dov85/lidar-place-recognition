"""Sparse variants (fewer laser beams) of a 64-beam variant onto this machine, for the
"sparsity" evaluation: blob container -> Drive (fetch_from_azure.py) -> <data_root>/<name>/.

The names come from experiments.sparse_variant, the same names azure_build.sh was given.

  python scripts/fetch_sparse.py --method ndt --variant 2026-10-04_baseline_717e81 \
      --drive /content/drive/MyDrive/lpr --data_root /content/ndt --sas_url "$AZURE_SAS"
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from lpr import evals  # noqa: E402
from lpr import experiments as ex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["ndt", "bev"])
    ap.add_argument("--variant", required=True, help="the 64-beam variant")
    ap.add_argument("--drive", required=True)
    ap.add_argument("--data_root", required=True, help="e.g. /content/ndt")
    ap.add_argument("--sas_url", default=None, help="needed only for what is not on Drive yet")
    ap.add_argument("--beams", type=int, nargs="+",
                    default=evals.EVAL_TYPES["sparsity"]["params"]["beams"])
    ap.add_argument("--seqs", nargs="+", default=evals.ALL_SEQS)
    a = ap.parse_args()
    base_cfg = os.path.join(a.drive, a.method, a.variant, "config.json")
    params = json.load(open(base_cfg))["params"]
    for b in a.beams:
        name, _ = ex.sparse_variant(a.method, a.variant, params, b)
        vdir = os.path.join(a.drive, a.method, name)
        local = os.path.join(a.data_root, name)
        tar = lambda s: os.path.join(vdir, ex.archive_name(a.method, s))  # noqa: E731
        missing = [s for s in a.seqs if not os.path.exists(tar(s))]
        if missing and a.sas_url:
            res = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "fetch_from_azure.py"),
                                  "--sas_url", a.sas_url, "--method", a.method, "--variant", name,
                                  "--seqs", *missing, "--drive", a.drive],
                                 capture_output=True, text=True)
            print(res.stdout.strip(), res.stderr[-1500:].strip())
        ext = "000000.png" if a.method == "bev" else "000000.bin"
        for s in a.seqs:
            if os.path.exists(tar(s)) and not os.path.exists(os.path.join(local, s, ext)):
                os.makedirs(local, exist_ok=True)
                subprocess.run(["tar", "-xf", tar(s), "-C", local], check=True)
        have = [s for s in a.seqs if os.path.exists(os.path.join(local, s, ext))]
        print(f"{b} beams ({name}): on this machine for {have}"
              + ("" if len(have) == len(a.seqs) else
                 f"; missing {sorted(set(a.seqs) - set(have))} (build on Azure: "
                 f"METHOD={a.method} VARIANT={name} ARGS=\"--beams {b}\" ...)"))


if __name__ == "__main__":
    main()
