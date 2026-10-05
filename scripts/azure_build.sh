#!/usr/bin/env bash
# Build the NDT cells (METHOD=ndt) or BEV images (METHOD=bev) of several KITTI sequences
# on an Azure VM and upload them, together with the raw scans, to an Azure Blob container.
#
# Run from inside a clone of this repository on the VM:
#   METHOD=bev VARIANT=<name> bash scripts/azure_build.sh "<container SAS URL>"
#
# The SAS URL is the container URL with a SAS token (read, add, create, write, list):
#   https://<account>.blob.core.windows.net/<container>?sv=...&sig=...
#
# Settings (environment variables, all optional except VARIANT):
#   METHOD     ndt (default) or bev
#   VARIANT    variant name, exactly as printed by the notebook (MyDrive/lpr/<METHOD>/<VARIANT>)
#   ARGS       options of that variant for extract_<METHOD>.py; empty = defaults
#              (NDT_ARGS is accepted for ndt, as before)
#   CELL_SEQS  sequences to build             (default: ndt 02 05 06 08, bev 00 02 05 06 08)
#   RAW_SEQS   sequences whose raw scans are kept in the blob container
#              (default: 00 02 05 06 08)
#   WORK       working folder on the data disk  (default: $HOME/work)
#
# Blob layout written by this script:
#   raw/kitti_seq<seq>_raw.tar             raw scans + poses + calib of one sequence
#   ndt/<VARIANT>/cells_<seq>.tar          NDT cells of one sequence
#   bev/<VARIANT>/imgs_<seq>.tar           BEV images of one sequence
#   <METHOD>/<VARIANT>/stats_<seq>.csv     per-scan statistics
#   <METHOD>/<VARIANT>/config_azure.json   parameters + per-sequence info (merged into Drive by Colab)
# Raw scans already in the container are reused, so a new variant never downloads KITTI again.
set -euo pipefail

SAS_URL="${1:?usage: METHOD=ndt|bev VARIANT=<name> bash scripts/azure_build.sh \"<container SAS URL>\"}"
METHOD="${METHOD:-ndt}"
case "$METHOD" in
  ndt) ARCHIVE=cells; DEFAULT_SEQS="02 05 06 08" ;;
  bev) ARCHIVE=imgs;  DEFAULT_SEQS="00 02 05 06 08" ;;
  *) echo "METHOD must be ndt or bev"; exit 1 ;;
esac
: "${VARIANT:?set VARIANT to the variant name printed by the notebook, e.g. VARIANT=2026-10-04_baseline_717e81}"
ARGS="${ARGS:-${NDT_ARGS:-}}"
CELL_SEQS="${CELL_SEQS:-$DEFAULT_SEQS}"
RAW_SEQS="${RAW_SEQS:-00 02 05 06 08}"
WORK="${WORK:-$HOME/work}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${SAS_URL%%\?*}"
TOKEN="${SAS_URL#*\?}"
blob_url() { echo "$BASE/$1?$TOKEN"; }
blob_exists() { [ "$(curl -s -o /dev/null -w '%{http_code}' -I "$(blob_url "$1")")" = "200" ]; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p "$WORK"
exec > >(tee -a "$WORK/build.log") 2>&1
log "method=$METHOD variant=$VARIANT seqs=[$CELL_SEQS] raw_seqs=[$RAW_SEQS] args=[$ARGS]"

# ---------------------------------------------------------------- tools
if [ ! -x "$WORK/venv/bin/python" ]; then
  log "installing python packages"
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip git curl >/dev/null
  python3 -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q numpy scipy scikit-learn pandas matplotlib requests remotezip pypatchworkpp
fi
PY="$WORK/venv/bin/python"
if [ "$METHOD" = bev ] && ! "$PY" -c "import cv2" 2>/dev/null; then
  log "installing opencv"
  "$WORK/venv/bin/pip" install -q opencv-python-headless
fi
if ! command -v azcopy >/dev/null; then
  log "installing azcopy"
  curl -sL https://aka.ms/downloadazcopy-v10-linux -o /tmp/azcopy.tgz
  tar -xzf /tmp/azcopy.tgz -C /tmp
  sudo cp /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/
fi
blob_put() { azcopy copy "$1" "$(blob_url "$2")" --overwrite=true --log-level=ERROR >/dev/null; }
blob_get() { azcopy copy "$(blob_url "$1")" "$2" --overwrite=true --log-level=ERROR >/dev/null; }

KITTI="$WORK/kitti"
OUT="$WORK/$METHOD/$VARIANT"
mkdir -p "$KITTI" "$OUT" "$WORK/tmp"

# ---------------------------------------------------------------- raw scans
have_raw() { [ -d "$KITTI/sequences/$1/velodyne" ] && [ -f "$KITTI/poses/$1.txt" ]; }
for seq in $RAW_SEQS $CELL_SEQS; do
  have_raw "$seq" && continue
  if blob_exists "raw/kitti_seq${seq}_raw.tar"; then
    log "seq $seq: raw scans from the blob container"
    blob_get "raw/kitti_seq${seq}_raw.tar" "$WORK/tmp/raw_$seq.tar"
    tar -xf "$WORK/tmp/raw_$seq.tar" -C "$KITTI" && rm -f "$WORK/tmp/raw_$seq.tar"
  else
    log "seq $seq: downloading raw scans from KITTI"
    "$PY" "$REPO/scripts/download_kitti.py" --out "$KITTI" --seqs "$seq" --threads 32
    log "seq $seq: uploading raw scans to the blob container"
    tar -cf "$WORK/tmp/raw_$seq.tar" -C "$KITTI" "sequences/$seq" "poses/$seq.txt"
    blob_put "$WORK/tmp/raw_$seq.tar" "raw/kitti_seq${seq}_raw.tar"
    rm -f "$WORK/tmp/raw_$seq.tar"
  fi
done

# ---------------------------------------------------------------- NDT cells / BEV images
for seq in $CELL_SEQS; do
  if blob_exists "$METHOD/$VARIANT/${ARCHIVE}_$seq.tar"; then
    log "seq $seq: already in the blob container, skipping"
    continue
  fi
  log "seq $seq: extract_$METHOD on $(ls "$KITTI/sequences/$seq/velodyne" | wc -l) scans, $(nproc) workers"
  # shellcheck disable=SC2086
  "$PY" "$REPO/scripts/extract_$METHOD.py" --kitti_root "$KITTI" --seq "$seq" --out_dir "$OUT" \
      --variant "$VARIANT" --workers "$(nproc)" $ARGS
  tar -cf "$WORK/tmp/${ARCHIVE}_$seq.tar" -C "$OUT" "$seq" "${seq}_stats.csv" config.json
  blob_put "$WORK/tmp/${ARCHIVE}_$seq.tar" "$METHOD/$VARIANT/${ARCHIVE}_$seq.tar"
  blob_put "$OUT/${seq}_stats.csv" "$METHOD/$VARIANT/stats_$seq.csv"
  blob_put "$OUT/config.json" "$METHOD/$VARIANT/config_azure.json"
  rm -f "$WORK/tmp/${ARCHIVE}_$seq.tar"
  log "seq $seq: uploaded"
done

log "done. Files of this variant in the blob container:"
for seq in $CELL_SEQS; do
  for f in "${ARCHIVE}_$seq.tar" "stats_$seq.csv"; do
    if blob_exists "$METHOD/$VARIANT/$f"; then echo "  ok      $METHOD/$VARIANT/$f"; else echo "  MISSING $METHOD/$VARIANT/$f"; fi
  done
done
blob_exists "$METHOD/$VARIANT/config_azure.json" && echo "  ok      $METHOD/$VARIANT/config_azure.json"
df -h "$WORK" | tail -1
