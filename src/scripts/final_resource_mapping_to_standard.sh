#!/usr/bin/env bash
# mapping_to_mni_parallel.sh
#
# This script maps both the rsfMRI 4D image and the aparc+aseg label file
# into MNI152_T1_2mm standard space using existing warp fields.
#
# USAGE:
#   ./mapping_to_mni_parallel.sh <FINAL_DIR> [--threads N]

set -euo pipefail

# -------- CLI Parsing --------
FINAL_DIR=$(realpath "$1"); shift
THREADS=4
if [[ "${1:-}" == --threads ]]; then
  THREADS="$2"; shift 2
fi

# -------- Init Log --------
LOG="$FINAL_DIR/mapping_to_mni.log"
: > "$LOG"
START_TIME=$(date +%s)

# -------- Paths --------
MNI_REF="$FSLDIR/data/standard/MNI152_T1_2mm"

# -------- Worker --------
worker() {
  DIR="$1"
  SUBJ=$(basename "$DIR")
  TMP=$(mktemp)
  log() { echo "$(date '+%F %T')  $SUBJ  $*" | tee -a "$TMP"; }

  FUNC="$DIR/${SUBJ}-rsfmri.nii.gz"
  LABEL="$DIR/${SUBJ}_aparc+aseg.nii.gz"
  FUNC_STD="$DIR/${SUBJ}-rsfmri_in_std.nii.gz"
  LABEL_STD="$DIR/${SUBJ}_aparc+aseg_in_std.nii.gz"

  # FEAT registration
  REG_DIR=("$FINAL_DIR/../resource/$SUBJ"/*.feat/reg)
  if [[ ${#REG_DIR[@]} -eq 0 ]]; then
    log "❌  reg directory missing"; cat "$TMP" >> "$LOG"; rm "$TMP"; return
  fi
  REG="${REG_DIR[0]}"

  WARP_FUNC2STD="$REG/example_func2standard_warp.nii.gz"

  # Check inputs
  if [[ ! -f "$FUNC" || ! -f "$LABEL" || ! -f "$WARP_FUNC2STD" ]]; then
    log "❌  missing input(s)"; cat "$TMP" >> "$LOG"; rm "$TMP"; return
  fi

  # Step 1: Warp full rsfMRI to standard
  if [[ ! -f "$FUNC_STD" ]]; then
    log "Warping rsfMRI to standard"
    applywarp -i "$FUNC" -r "$MNI_REF" -w "$WARP_FUNC2STD" -o "$FUNC_STD" >> "$TMP" 2>&1
  else
    log "✅ rsfMRI in std exists"
  fi

  # Step 2: Warp label to standard
  if [[ ! -f "$LABEL_STD" ]]; then
    log "Warping label to standard"
    applywarp -i "$LABEL" -r "$MNI_REF" -w "$WARP_FUNC2STD" --interp=nn -o "$LABEL_STD" >> "$TMP" 2>&1
  else
    log "✅ label in std exists"
  fi

  log "✅ Done"
  { echo "─────────── $SUBJ ───────────"; cat "$TMP"; echo; } | tee -a "$LOG"
  rm "$TMP"
}

export -f worker
export FINAL_DIR LOG MNI_REF

# -------- Run in parallel --------
find "$FINAL_DIR" -mindepth 1 -maxdepth 1 -type d -print0 \
  | xargs -0 -n1 -P "$THREADS" bash -c 'worker "$0"'

# -------- Timing Summary --------
END_TIME=$(date +%s)
DURATION=$(( END_TIME - START_TIME ))
HOURS=$(echo "scale=2; $DURATION / 3600" | bc)
echo "$(date '+%F %T')  All Done. Total time: $HOURS hours." | tee -a "$LOG"

