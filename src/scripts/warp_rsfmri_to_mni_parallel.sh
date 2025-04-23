#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# warp_rsfmri_parallel.sh
#
# Usage:
#   ./warp_rsfmri_parallel.sh <final_filtered_root> --jobs <n_jobs>
#
# Example:
#   ./warp_rsfmri_parallel.sh \
#     /export/kroy/Capstone_Project-fMRI_analysis/resources/final_filtered_resource \
#     --jobs 12
# -------------------------------------------------------------------

usage() {
  echo "Usage: $0 <final_filtered_root> --jobs <n_jobs>"
  exit 1
}

# Require exactly 3 args: directory, --jobs, N
if [ $# -ne 3 ] || [ "$2" != "--jobs" ]; then
  usage
fi

FINAL_ROOT="$1"       # e.g. /…/final_filtered_resource
NJOBS="$3"            # e.g. 12

# Master log file
LOG_FILE="$FINAL_ROOT/warp_rsfmri_parallel.log"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

# Check for GNU parallel
if ! command -v parallel &> /dev/null; then
  echo "ERROR: GNU parallel is required but not found."
  exit 1
fi

# Ensure FSLDIR is set
if [ -z "${FSLDIR:-}" ]; then
  echo "ERROR: FSLDIR is not set. Please source your FreeSurfer/FSL setup first."
  exit 1
fi

MNI_REF="$FSLDIR/data/standard/MNI152_T1_2mm"
RESOURCE_ROOT="$(dirname "$FINAL_ROOT")/resource"

warp_rsfmri() {
  local SUBJ="$1"
  local FINAL_DIR="$FINAL_ROOT/$SUBJ"
  local REG_DIR="$RESOURCE_ROOT/$SUBJ/prestats_${SUBJ}.feat/reg"

  local FUNC4D="$FINAL_DIR/${SUBJ}-rsfmri.nii.gz"
  local MEAN3D="$FINAL_DIR/${SUBJ}-rsfmri_mean.nii.gz"
  local WARPDEF="$REG_DIR/example_func2standard_warp.nii.gz"
  local OUT_4D="$FINAL_DIR/${SUBJ}-rsfmri_4D_in_MNI.nii.gz"

  # Buffer this subject’s entire log
  local TMPLOG
  TMPLOG="$(mktemp "${TMPDIR:-/tmp}/warp_${SUBJ}.XXXXXX")"

  {
    echo "----------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $SUBJ"
    echo

    # (Optional) generate mean EPI
    fslmaths "$FUNC4D" -Tmean "$MEAN3D" \
      && echo "  • Mean EPI: $(basename "$MEAN3D")" \
      || echo "  !! Mean EPI FAILED"

    # Warp full 4D → MNI
    applywarp \
      -i "$FUNC4D" \
      -r "$MNI_REF" \
      -w "$WARPDEF" \
      -o "$OUT_4D" \
      && echo "  • 4D warp: $(basename "$OUT_4D")" \
      || echo "  !! 4D warp FAILED"

    echo
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] END   $SUBJ"
    echo "----------------------------------------"
  } &> "$TMPLOG"

  # Append atomically to the master log and echo to stdout
  cat "$TMPLOG" | tee -a "$LOG_FILE"
  rm -f "$TMPLOG"
}

export FINAL_ROOT RESOURCE_ROOT MNI_REF LOG_FILE
export -f warp_rsfmri

# Run in parallel, preserving input order via --keep-order
find "$FINAL_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
  | parallel --keep-order --jobs "$NJOBS" warp_rsfmri {}

