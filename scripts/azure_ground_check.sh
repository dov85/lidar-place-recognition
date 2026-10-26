#!/usr/bin/env bash
# Ground check (scripts/ground_coverage.py: NDT cells with and without ground removal) of
# several sequences on the Azure VM. Raw scans: $WORK/kitti, or raw/kitti_seq<seq>_raw.tar
# from the blob container (as azure_build.sh).
#
# Run from inside the clone of this repository on the VM (in tmux):
#   bash scripts/azure_ground_check.sh "<container SAS URL>"
#
# Settings (environment variables):
#   SEQS     sequences (default: 00 02 05 06 08)
#   N_SCANS  random scans per sequence (default 300, as notebook 06; 0 = all)
#   PARAMS   JSON, NDT parameters that differ from the defaults (default {})
#   WORK     working folder (default: $HOME/work)
#
# Results, in the blob container (notebook 06, section 7, copies them to Drive):
#   ground_check/<TAG>/ground_coverage_<seq>.csv, _summary.csv, _params.json,
#   ground_check/<TAG>/ground_previews_<seq>.tar
# TAG = lpr.experiments.ground_check_tag(PARAMS, N_SCANS), the folder name of notebook 06.
# A sequence whose summary is already in the blob container is skipped.
set -euo pipefail

SAS_URL="${1:?usage: bash scripts/azure_ground_check.sh \"<container SAS URL>\"}"
SEQS="${SEQS:-00 02 05 06 08}"
N_SCANS="${N_SCANS:-300}"
PARAMS="${PARAMS:-}"
[ -n "$PARAMS" ] || PARAMS="{}"
WORK="${WORK:-$HOME/work}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${SAS_URL%%\?*}"
TOKEN="${SAS_URL#*\?}"
blob_url() { echo "$BASE/$1?$TOKEN"; }
blob_exists() { [ "$(curl -s -o /dev/null -w '%{http_code}' -I "$(blob_url "$1")")" = "200" ]; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

mkdir -p "$WORK/tmp"
exec > >(tee -a "$WORK/ground_check.log") 2>&1

if [ ! -x "$WORK/venv/bin/python" ]; then
  log "installing python packages"
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv python3-pip git curl >/dev/null
  python3 -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q numpy scipy scikit-learn pandas matplotlib requests remotezip pypatchworkpp
fi
PY="$WORK/venv/bin/python"
"$PY" -c "import matplotlib, pandas" 2>/dev/null || "$WORK/venv/bin/pip" install -q matplotlib pandas
if ! command -v azcopy >/dev/null; then
  log "installing azcopy"
  curl -sL https://aka.ms/downloadazcopy-v10-linux -o /tmp/azcopy.tgz
  tar -xzf /tmp/azcopy.tgz -C /tmp
  sudo cp /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/
fi
blob_put() { azcopy copy "$1" "$(blob_url "$2")" --overwrite=true --log-level=ERROR >/dev/null; }
blob_get() { azcopy copy "$(blob_url "$1")" "$2" --overwrite=true --log-level=ERROR >/dev/null; }

TAG="$(cd "$REPO" && "$PY" -c "import sys, json; sys.path.insert(0, 'src'); \
from lpr.experiments import ground_check_tag; print(ground_check_tag(json.loads(sys.argv[1]), $N_SCANS))" "$PARAMS")"
KITTI="$WORK/kitti"
OUT="$WORK/ground_check/$TAG"
mkdir -p "$KITTI" "$OUT"
log "ground check tag=$TAG seqs=[$SEQS] n_scans=$N_SCANS params=$PARAMS, $(nproc) cores"

for seq in $SEQS; do
  if blob_exists "ground_check/$TAG/ground_coverage_${seq}_summary.csv"; then
    log "seq $seq: already in the blob container, skipping"
    continue
  fi
  if [ ! -d "$KITTI/sequences/$seq/velodyne" ]; then
    if ! blob_exists "raw/kitti_seq${seq}_raw.tar"; then
      log "seq $seq: no raw scans on the VM or in the blob container - skipped"
      continue
    fi
    log "seq $seq: raw scans from the blob container"
    blob_get "raw/kitti_seq${seq}_raw.tar" "$WORK/tmp/raw_$seq.tar"
    tar -xf "$WORK/tmp/raw_$seq.tar" -C "$KITTI" && rm -f "$WORK/tmp/raw_$seq.tar"
  fi
  log "seq $seq: cells with and without ground removal"
  "$PY" "$REPO/scripts/ground_coverage.py" --kitti_root "$KITTI" --seq "$seq" --n_scans "$N_SCANS" \
      --out_dir "$OUT" --params "$PARAMS" --workers "$(nproc)" | { grep -v -i "patchworkpp::" || true; }
  tar -cf "$WORK/tmp/ground_previews_$seq.tar" -C "$OUT" "ground_previews_$seq"
  blob_put "$WORK/tmp/ground_previews_$seq.tar" "ground_check/$TAG/ground_previews_$seq.tar"
  rm -f "$WORK/tmp/ground_previews_$seq.tar"
  for f in "ground_coverage_$seq.csv" "ground_coverage_${seq}_params.json" "ground_coverage_${seq}_summary.csv"; do
    blob_put "$OUT/$f" "ground_check/$TAG/$f"     # summary last: it marks the sequence as done
  done
  log "seq $seq: uploaded to ground_check/$TAG/"
done
log "done"
