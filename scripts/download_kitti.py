"""Download only the parts of KITTI Odometry that are needed.

The Velodyne archive is ~80 GB. Instead of downloading all of it, the scans of
the requested sequences are read straight out of the remote zip with HTTP range
requests (remotezip), in parallel. The poses archive is small and is downloaded
in full.

Example:
  python scripts/download_kitti.py --out /content/kitti --seqs 00
"""
import argparse
import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests
from remotezip import RemoteZip

BASE = os.environ.get("KITTI_BASE_URL", "https://s3.eu-central-1.amazonaws.com/avg-kitti/")
VELODYNE_ZIP = BASE + "data_odometry_velodyne.zip"
POSES_ZIP = BASE + "data_odometry_poses.zip"
CALIB_ZIP = BASE + "data_odometry_calib.zip"


def small_zip(url, out, members_prefix=None):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            if members_prefix and not any(name.startswith(p) for p in members_prefix):
                continue
            if name.endswith("/"):
                continue
            rel = name.split("dataset/", 1)[-1]
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(name) as src, open(dst, "wb") as f:
                f.write(src.read())


def fetch_chunk(names, out):
    done = 0
    with RemoteZip(VELODYNE_ZIP) as z:
        for name in names:
            rel = name.split("dataset/", 1)[-1]
            dst = os.path.join(out, rel)
            if os.path.exists(dst) and os.path.getsize(dst) == z.getinfo(name).file_size:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            for attempt in range(5):
                try:
                    data = z.read(name)
                    break
                except Exception as e:  # network hiccup: retry
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
            with open(dst + ".tmp", "wb") as f:
                f.write(data)
            os.replace(dst + ".tmp", dst)
            done += 1
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqs", nargs="+", default=["00"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--first", type=int, default=None, help="only the first N scans per sequence")
    ap.add_argument("--poses_only", action="store_true", help="skip the Velodyne scans")
    args = ap.parse_args()

    print("poses + calib ...", flush=True)
    small_zip(POSES_ZIP, args.out)
    small_zip(CALIB_ZIP, args.out,
              members_prefix=[f"dataset/sequences/{s}/" for s in args.seqs])

    if args.poses_only:
        print("poses ->", os.path.join(args.out, "poses"))
        return
    print("reading the Velodyne zip index ...", flush=True)
    with RemoteZip(VELODYNE_ZIP) as z:
        names = z.namelist()
    todo = []
    for s in args.seqs:
        seq_names = sorted(n for n in names
                           if n.startswith(f"dataset/sequences/{s}/velodyne/") and n.endswith(".bin"))
        if args.first:
            seq_names = seq_names[:args.first]
        print(f"sequence {s}: {len(seq_names)} scans", flush=True)
        todo += seq_names

    chunks = [todo[i::args.threads] for i in range(args.threads)]
    t0 = time.time()
    with ThreadPoolExecutor(args.threads) as ex:
        n = sum(ex.map(lambda c: fetch_chunk(c, args.out), chunks))
    print(f"downloaded {n} new scans in {(time.time() - t0) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
