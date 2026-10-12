"""Upload one file to the Azure blob container (block blob, 32 MB blocks, retries).

Used by notebook 03 to put cells built in Colab (cells_00.tar) where the VM can read them.
The SAS URL needs write + create permission (the project's SAS has read, add, create,
write, list).

Example:
  python scripts/upload_to_azure.py --sas_url "<container SAS URL>" \
      --src /content/drive/MyDrive/lpr/ndt/<variant>/cells_00.tar --dest ndt/<variant>/cells_00.tar
"""
import argparse
import base64
import os
import time

import requests

BLOCK = 32 << 20
HEADERS = {"x-ms-version": "2021-08-06"}


def blob_url(sas_url, path, extra=""):
    base, token = sas_url.split("?", 1)
    return f"{base.rstrip('/')}/{path}?{token}{extra}"


def exists(sas_url, path):
    try:
        code = requests.head(blob_url(sas_url, path), headers=HEADERS, timeout=60).status_code
    except requests.RequestException as e:   # str(e) would contain the SAS token
        raise SystemExit(f"{path}: network error ({type(e).__name__})") from None
    if code in (401, 403):
        raise SystemExit(f"HTTP {code} when reading {path}: the SAS URL is not accepted "
                         "(expired, mistyped, or without read permission). Create a new one in the "
                         "Azure portal (container -> Shared access tokens; set the expiry weeks "
                         "ahead, the portal proposes a few hours) and update AZURE_SAS.")
    return code == 200


def _put(url, data, headers, tries=5):
    for t in range(tries):
        try:
            r = requests.put(url, data=data, headers={**HEADERS, **headers}, timeout=300)
            if r.status_code in (200, 201):
                return
            msg = f"HTTP {r.status_code}: {r.text[:300]}"
            if 400 <= r.status_code < 500:   # not a network problem: retrying does not help
                raise SystemExit(f"upload refused, {msg}")
        except requests.RequestException as e:
            msg = f"network error ({type(e).__name__})"   # str(e) would contain the SAS token
        print(f"  retry {t + 1}: {msg}", flush=True)
        time.sleep(5 * (t + 1))
    raise SystemExit("upload failed")


def upload(sas_url, src, dest):
    size = os.path.getsize(src)
    ids, t0 = [], time.time()
    with open(src, "rb") as f:
        i = 0
        while True:
            chunk = f.read(BLOCK)
            if not chunk:
                break
            bid = base64.b64encode(f"{i:08d}".encode()).decode()
            _put(blob_url(sas_url, dest, f"&comp=block&blockid={requests.utils.quote(bid)}"),
                 chunk, {})
            ids.append(bid)
            i += 1
            print(f"  {min(i * BLOCK, size) / 2**20:.0f} / {size / 2**20:.0f} MB", end="\r",
                  flush=True)
    body = "<?xml version='1.0' encoding='utf-8'?><BlockList>" + \
        "".join(f"<Latest>{b}</Latest>" for b in ids) + "</BlockList>"
    _put(blob_url(sas_url, dest, "&comp=blocklist"), body.encode(),
         {"Content-Type": "application/xml"})
    print(f"\n  {os.path.basename(src)} -> {dest}: {size / 2**20:.0f} MB in "
          f"{time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sas_url", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True, help="path inside the container")
    ap.add_argument("--force", action="store_true", help="upload even if the blob exists")
    a = ap.parse_args()
    if not a.force and exists(a.sas_url, a.dest):
        print(f"{a.dest}: already in the blob container")
        return
    upload(a.sas_url, a.src, a.dest)


if __name__ == "__main__":
    main()
