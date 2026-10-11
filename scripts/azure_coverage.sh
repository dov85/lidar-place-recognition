#!/usr/bin/env bash
# Checks (coverage + pictures, scripts/ndt_coverage.py) of an NDT variant that was built before
# the checks were part of extract_ndt.py, on the Azure VM, for all its sequences.
#
# Run from inside the clone of this repository on the VM (in tmux):
#   VARIANT=<name> bash scripts/azure_coverage.sh "<container SAS URL>"
#
# Settings (environment variables):
#   VARIANT  variant name (MyDrive/lpr/ndt/<VARIANT>), required
#   SEQS     sequences (default: 00 02 05 06 08)
#   EVERY    check every n-th scan (default 1 = all)
#   WORK     working folder (default: $HOME/work, as azure_build.sh)
#
# Uses the cells already on the VM disk ($WORK/ndt/<VARIANT>/<seq>/), otherwise downloads
# ndt/<VARIANT>/cells_<seq>.tar from the blob container (notebook 03 uploads cells_00.tar,
# which was built in Colab). Raw scans: $WORK/kitti or raw/kitti_seq<seq>_raw.tar.
# Results, in the blob container (notebook 03 copies them to Drive):
#   ndt/<VARIANT>/coverage/coverage_<seq>.csv, coverage_<seq>_summary.json, previews_<seq>.tar
set -euo pipefail

SAS_URL="${1:?usage: VARIANT=<name> bash scripts/azure_coverage.sh \"<container SAS URL>\"}"
: "${VARIANT:?set VARIANT to the variant name, e.g. VARIANT=2026-10-04_baseline_717e81}"
SEQS="${SEQS:-00 02 05 06 08}"
EVERY="${EVERY:-1}"
WORK="${WORK:-$HOME/work}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${SAS_URL%%\?*}"
TOKEN="${SAS_URL#*\?}"
blob_url() { echo "$BASE/$1?$TOKEN"; }
blob_exists() { [ "$(curl -s -o /dev/null -w '%{http_code}' -I "$(blob_url "$1")")" = "200" ]; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p "$WORK/tmp"
exec > >(tee -a "$WORK/coverage.log") 2>&1
log "coverage of variant=$VARIANT seqs=[$SEQS] every=$EVERY"

if [ ! -x "$WORK/venv/bin/python" ]; then
  log "installing python packages"
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip git curl >/dev/null
  python3 -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q numpy scipy scikit-learn pandas matplotlib requests remotezip pypatchworkpp
fi
PY="$WORK/venv/bin/python"
"$PY" -c "import matplotlib" 2>/dev/null || "$WORK/venv/bin/pip" install -q matplotlib
if ! command -v azcopy >/dev/null; then
  log "installing azcopy"
  curl -sL https://aka.ms/downloadazcopy-v10-linux -o /tmp/azcopy.tgz
  tar -xzf /tmp/azcopy.tgz -C /tmp
  sudo cp /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/
fi
blob_put() { azcopy copy "$1" "$(blob_url "$2")" --overwrite=true --log-level=ERROR >/dev/null; }
blob_get() { azcopy copy "$(blob_url "$1")" "$2" --overwrite=true --log-level=ERROR >/dev/null; }

KITTI="$WORK/kitti"
NDT="$WORK/ndt/$VARIANT"
OUT="$WORK/coverage/$VARIANT"
mkdir -p "$KITTI" "$NDT" "$OUT"

for seq in $SEQS; do
  if [ ! -d "$KITTI/sequences/$seq/velodyne" ]; then
    if ! blob_exists "raw/kitti_seq${seq}_raw.tar"; then
      log "seq $seq: no raw scans on the VM or in the blob container - skipped"
      continue
    fi
    log "seq $seq: raw scans from the blob container"
    blob_get "raw/kitti_seq${seq}_raw.tar" "$WORK/tmp/raw_$seq.tar"
    tar -xf "$WORK/tmp/raw_$seq.tar" -C "$KITTI" && rm -f "$WORK/tmp/raw_$seq.tar"
  fi
  if [ ! -f "$NDT/$seq/000000.bin" ] || [ ! -f "$NDT/config.json" ]; then
    if ! blob_exists "ndt/$VARIANT/cells_$seq.tar"; then
      log "seq $seq: no cells on the VM and no ndt/$VARIANT/cells_$seq.tar in the blob container"
      log "         (for 00: run section 3 of notebooks/03_ndt_checks.ipynb first) - skipped"
      continue
    fi
    log "seq $seq: cells from the blob container"
    blob_get "ndt/$VARIANT/cells_$seq.tar" "$WORK/tmp/cells_$seq.tar"
    mkdir -p "$WORK/tmp/cells_$seq"
    tar -xf "$WORK/tmp/cells_$seq.tar" -C "$WORK/tmp/cells_$seq"
    mv "$WORK/tmp/cells_$seq/$seq" "$NDT/$seq"
    [ -f "$NDT/config.json" ] || cp "$WORK/tmp/cells_$seq/config.json" "$NDT/config.json"
    rm -rf "$WORK/tmp/cells_$seq" "$WORK/tmp/cells_$seq.tar"
  fi
  log "seq $seq: checking"
  "$PY" "$REPO/scripts/ndt_coverage.py" --kitti_root "$KITTI" --ndt_dir "$NDT" --seq "$seq" \
      --out_dir "$OUT" --every "$EVERY" --workers "$(nproc)" | grep -v -i patchwork
  tar -cf "$WORK/tmp/previews_$seq.tar" -C "$OUT" "previews_$seq"
  blob_put "$OUT/coverage_$seq.csv" "ndt/$VARIANT/coverage/coverage_$seq.csv"
  blob_put "$OUT/coverage_${seq}_summary.json" "ndt/$VARIANT/coverage/coverage_${seq}_summary.json"
  blob_put "$WORK/tmp/previews_$seq.tar" "ndt/$VARIANT/coverage/previews_$seq.tar"
  rm -f "$WORK/tmp/previews_$seq.tar"
  log "seq $seq: uploaded"
done
log "done. Copy the results to Drive with section 5 of notebooks/03_ndt_checks.ipynb"
