"""Read-only access to the project folder through a public Google Drive link.

The folder MyDrive/lpr (variants, runs, summaries) can be shared as "anyone with the link". With
that link, files are fetched by their path inside the folder and copied into a local folder with
the same layout (the "mirror"), so the rest of the code sees an ordinary DRIVE folder:

  mirror_configs(url, drive)                  every variant and run config.json (small), so that
                                              resolve_variant / resolve_run find the existing names
  fetch(url, "ndt/<variant>/cells_08.tar", drive)          one file, if it is not there yet
  fetch_checkpoint(url, drive, "myndt", run, "last")       the checkpoint of a run

Folders are listed with the public embedded folder view (no Google account or API key); files are
downloaded with gdown (preinstalled on Colab), which handles the confirmation page of large files.
"""
import functools
import html
import json
import os
import re

import requests

ENTRY = re.compile(r'<div class="flip-entry" id="entry-([^"]+)"[\s\S]*?<a href="([^"]+)"'
                   r'[\s\S]*?<div class="flip-entry-title">([^<]*)</div>')


def folder_id(url):
    """Id of a Drive folder from its share link (or the id itself)."""
    m = re.search(r"/folders/([A-Za-z0-9_-]+)", url) or re.search(r"[?&]id=([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url):
        return url
    raise ValueError(f"not a Google Drive folder link: {url}")


@functools.lru_cache(maxsize=None)
def list_folder(fid):
    """{name: (id, is_folder)} of the items of a public folder."""
    r = requests.get("https://drive.google.com/embeddedfolderview", params={"id": fid}, timeout=60)
    r.raise_for_status()
    return {html.unescape(name): (i, "/folders/" in href) for i, href, name in ENTRY.findall(r.text)}


def resolve(url, relpath):
    """(id, is_folder) of a path inside the shared folder, or None if it is not there."""
    fid, is_dir = folder_id(url), True
    for part in [p for p in relpath.strip("/").split("/") if p]:
        if not is_dir:
            return None
        item = list_folder(fid).get(part)
        if item is None:
            return None
        fid, is_dir = item
    return fid, is_dir


def listdir(url, relpath):
    """Names in a folder of the shared folder ([] if it is not there)."""
    item = resolve(url, relpath)
    return sorted(list_folder(item[0])) if item and item[1] else []


def fetch(url, relpath, local_root, quiet=True):
    """Copy one file of the shared folder to <local_root>/<relpath> (if it is not there yet).
    Returns the local path, or None if the shared folder has no such file."""
    dest = os.path.join(local_root, relpath)
    if os.path.exists(dest):
        return dest
    item = resolve(url, relpath)
    if item is None or item[1]:
        return None
    import gdown
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    if not quiet:
        print(f"  {relpath} <- public Drive link")
    gdown.download(id=item[0], output=tmp, quiet=quiet)
    os.replace(tmp, dest)
    return dest


def mirror_configs(url, local_root, methods=("ndt", "bev", "myndt")):
    """Every variant and run config.json, and the summary tables, into the mirror. Small files;
    the cells, images and checkpoints are fetched later, only the ones that are needed."""
    n = 0
    for f in ("evals_summary.csv", "runs_summary.csv", "density_check.csv"):
        n += fetch(url, f, local_root) is not None
    for data in ("ndt", "bev"):
        for variant in listdir(url, data):
            n += fetch(url, f"{data}/{variant}/config.json", local_root) is not None
    for m in methods:
        for run in listdir(url, f"runs/{m}"):
            run_dir = f"runs/{m}/{run}"
            n += fetch(url, f"{run_dir}/config.json", local_root) is not None
            files = listdir(url, run_dir)                # remembered: which checkpoints exist there
            os.makedirs(os.path.join(local_root, run_dir), exist_ok=True)
            with open(os.path.join(local_root, run_dir, "public_files.json"), "w") as fh:
                json.dump(files, fh)
    print(f"public Drive link: {n} configuration files in {local_root}")
    return n


def checkpoints(local_run_dir):
    """Checkpoint files of a run, on this machine or (after mirror_configs) in the shared folder."""
    names = set(f for f in os.listdir(local_run_dir) if re.fullmatch(r"epoch_\d+\.pth", f)) \
        if os.path.isdir(local_run_dir) else set()
    listed = os.path.join(local_run_dir, "public_files.json")
    if os.path.exists(listed):
        names |= set(f for f in json.load(open(listed)) if re.fullmatch(r"epoch_\d+\.pth", f))
    return sorted(names, key=lambda f: int(re.search(r"\d+", f).group()))


def fetch_checkpoint(url, local_root, method, run, ckpt="last"):
    """The checkpoint of a run ("last" = the highest epoch) into the mirror; returns its path."""
    run_dir = os.path.join(local_root, "runs", method, run)
    if ckpt == "last":
        have = checkpoints(run_dir)
        if not have:
            return None
        ckpt = have[-1]
    if os.path.isabs(ckpt):
        return ckpt
    return fetch(url, f"runs/{method}/{run}/{ckpt}", local_root, quiet=False)
