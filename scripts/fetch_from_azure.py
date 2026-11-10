"""Copy a variant built on Azure (scripts/azure_build.sh) into its Drive folder.

Downloads <method>/<variant>/<archive>_<seq>.tar (cells_XX.tar for ndt, imgs_XX.tar for
bev) and stats_<seq>.csv from the blob container for every requested sequence that is
not on Drive yet, and merges the per-sequence information of config_azure.json into the
variant's config.json (after checking that the parameters are the same). A variant that
is not on Drive yet (a new one, built on Azure first) gets config_azure.json as its
config, after checking that the hash in its name matches the parameters.
With --rot_copies N (ndt): the point-rotated copies cells_<seq>_rot<N>.tar of the given
sequences instead (azure_build.sh with ROT_COPIES=N).
NDT variants built with the built-in checks also have previews_<seq>.tar (pictures of 10
scans); they are unpacked to <variant>/previews_<seq>/.

Example:
  python scripts/fetch_from_azure.py --sas_url "<container SAS URL>" --method ndt \
      --variant 2026-10-04_baseline_717e81 --seqs 02 05 06 08 --drive /content/drive/MyDrive/lpr
"""
import argparse
import json
import os
import sys
import tarfile
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lpr.experiments import (archive_name, rot_archive_name, rot_dir_name,  # noqa: E402
                             upgrade_ndt_config, write_json)


def blob_url(sas_url, path):
    base, token = sas_url.split("?", 1)
    return f"{base.rstrip('/')}/{path}?{token}"


SAS_HELP = ("the SAS URL is not accepted (expired, mistyped, or without the needed permission). "
            "Create a new one in the Azure portal (container -> Shared access tokens; the portal "
            "proposes an expiry of a few hours: set it weeks ahead) and update AZURE_SAS.")


def download(url, dest):
    """Never prints the URL: it contains the SAS token."""
    t0 = time.time()
    path = url.split("?", 1)[0].split(".net/", 1)[-1]
    try:
        r = requests.get(url, stream=True, timeout=60)
    except requests.RequestException as e:
        raise SystemExit(f"{path}: network error ({type(e).__name__})") from None
    with r:
        if r.status_code == 404:
            return False
        if r.status_code in (401, 403):
            raise SystemExit(f"HTTP {r.status_code} for {path}: {SAS_HELP}")
        if r.status_code >= 400:
            raise SystemExit(f"HTTP {r.status_code} for {path}")
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
        os.replace(tmp, dest)
    mb = os.path.getsize(dest) / 2**20
    print(f"  {os.path.basename(dest)}: {mb:.0f} MB in {time.time() - t0:.0f} s")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sas_url", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seqs", nargs="+", required=True)
    ap.add_argument("--drive", required=True)
    ap.add_argument("--method", default="ndt", choices=["ndt", "bev"])
    ap.add_argument("--rot_copies", type=int, default=0)
    a = ap.parse_args()

    vdir = os.path.join(a.drive, a.method, a.variant)
    os.makedirs(vdir, exist_ok=True)
    cfg_path = os.path.join(vdir, "config.json")
    prefix = f"{a.method}/{a.variant}"
    az_path = os.path.join(vdir, "config_azure.json")
    if not download(blob_url(a.sas_url, f"{prefix}/config_azure.json"), az_path):
        raise SystemExit("config_azure.json not in the blob container: run azure_build.sh first")
    az, _ = upgrade_ndt_config(json.load(open(az_path)))
    if os.path.exists(cfg_path):
        cfg, _ = upgrade_ndt_config(json.load(open(cfg_path)))
    else:
        if not a.variant.endswith("_" + az["params_hash"]):
            raise SystemExit(f"variant name {a.variant} does not end with the hash of the "
                             f"parameters built on Azure ({az['params_hash']})")
        cfg = {k: v for k, v in az.items() if k != "seqs"}
        cfg["variant"] = a.variant
        cfg["seqs"] = {}
    if az["params_hash"] != cfg["params_hash"]:
        raise SystemExit(f"parameters on Azure ({az['params_hash']}) differ from Drive "
                         f"({cfg['params_hash']}): the Azure run used other NDT_ARGS")

    if a.rot_copies:
        for seq in a.seqs:
            sub = rot_dir_name(seq, a.rot_copies)
            tar = os.path.join(vdir, rot_archive_name(seq, a.rot_copies))
            if not os.path.exists(tar):
                print(f"{sub}: downloading")
                if not download(blob_url(a.sas_url, f"{prefix}/{os.path.basename(tar)}"), tar):
                    print(f"{sub}: not in the blob container (build it with ROT_COPIES={a.rot_copies})")
                    continue
                download(blob_url(a.sas_url, f"{prefix}/stats_{sub}.csv"),
                         os.path.join(vdir, f"stats_{sub}.csv"))
            else:
                print(f"{sub}: already on Drive")
            if sub in az.get("rot_copies", {}):
                cfg.setdefault("rot_copies", {})[sub] = dict(az["rot_copies"][sub], built_on="azure")
        write_json(cfg_path, cfg)
        print("point-rotated copies in", a.variant + ":", sorted(cfg.get("rot_copies", {})))
        return

    for seq in a.seqs:
        tar = os.path.join(vdir, archive_name(a.method, seq))
        if os.path.exists(tar):
            print(f"seq {seq}: already on Drive")
            if seq in az.get("seqs", {}) and seq not in cfg.get("seqs", {}):
                cfg.setdefault("seqs", {})[seq] = dict(az["seqs"][seq], built_on="azure")
            continue
        print(f"seq {seq}: downloading")
        if not download(blob_url(a.sas_url, f"{prefix}/{archive_name(a.method, seq)}"), tar):
            print(f"seq {seq}: not in the blob container (not built yet?)")
            continue
        download(blob_url(a.sas_url, f"{prefix}/stats_{seq}.csv"),
                 os.path.join(vdir, f"stats_{seq}.csv"))
        if a.method == "ndt":
            unpack_previews(blob_url(a.sas_url, f"{prefix}/previews_{seq}.tar"), vdir, seq)
        if seq in az.get("seqs", {}):
            cfg.setdefault("seqs", {})[seq] = dict(az["seqs"][seq], built_on="azure")
    write_json(cfg_path, cfg)
    print("sequences in", a.variant + ":", sorted(cfg.get("seqs", {})))


def unpack_previews(url, dest_dir, seq):
    """previews_<seq>.tar (folder <seq>_previews/ or previews_<seq>/) -> dest_dir/previews_<seq>/"""
    tmp = os.path.join(dest_dir, f".previews_{seq}.tar")
    if not download(url, tmp):
        return False
    out = os.path.join(dest_dir, f"previews_{seq}")
    os.makedirs(out, exist_ok=True)
    with tarfile.open(tmp) as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith(".png"):
                with t.extractfile(m) as src, open(os.path.join(out, os.path.basename(m.name)),
                                                   "wb") as dst:
                    dst.write(src.read())
    os.remove(tmp)
    print(f"  pictures -> {out}")
    return True


if __name__ == "__main__":
    main()
