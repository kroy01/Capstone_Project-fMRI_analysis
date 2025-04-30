#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   build_all_graphs.sh \
#     --inp_path  <input_parent_dir> \
#     --out_path  <output_dir> \
#     [--roi_ids <id1,id2,...>] \
#     [--jobs   <num_jobs>]
#
# Example:
#   build_all_graphs.sh \
#     --inp_path /data/fmri \
#     --out_path /data/graphs \
#     --roi_ids 17,18,53 \
#     --jobs    4

# Defaults
JOBS=1
ROI_IDS=""
THRESHOLD=1e-6
CORR_THRESHOLD=0.3
MAX_DISTANCE=10.0
SCRIPT_PATH="/export/kroy/Capstone_Project-fMRI_analysis/gnn/src/create_fmri_bold_graph.py"

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --inp_path)
      INPUT_DIR="$2"; shift 2 ;;
    --out_path)
      OUTPUT_DIR="$2"; shift 2 ;;
    --roi_ids)
      ROI_IDS="$2"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required
if [ -z "${INPUT_DIR:-}" ] || [ -z "${OUTPUT_DIR:-}" ]; then
  echo "ERROR: --inp_path and --out_path are required."
  echo "Usage: $0 --inp_path <input_parent_dir> --out_path <output_dir> [--roi_ids <ids>] [--jobs <n>]"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOGFILE="${OUTPUT_DIR}/complete_graph_build.log"
: > "$LOGFILE"

process_subject() {
  subj_dir="$1"
  subj_id=$(basename "${subj_dir%/}")

  # infer numeric label from suffix
  suffix="${subj_id##*_}"
  case "$suffix" in
    CN)    graph_label=0 ;;
    EMCI)  graph_label=1 ;;
    MCI)   graph_label=2 ;;
    *)
      echo "⚠️  Unknown suffix '$suffix' for ${subj_id}, defaulting label→0"
      graph_label=0
      ;;
  esac

  fmri_path="${subj_dir}/${subj_id}-rsfmri_4D_in_MNI.nii.gz"
  label_path="${subj_dir}/${subj_id}_aparc+aseg.nii.gz"
  out_path="${OUTPUT_DIR}/${subj_id}_bold_complete.pt"

  tmp=$(mktemp)
  {
    echo "----- ${subj_id} START $(date '+%Y-%m-%d %H:%M:%S') -----"
    echo "Label inferred: ${suffix} → ${graph_label}"
    echo "Threshold: ${THRESHOLD}, Corr-thresh: ${CORR_THRESHOLD}, Max-dist: ${MAX_DISTANCE}"
    [ -n "$ROI_IDS" ] && echo "ROI filter: ${ROI_IDS}"

    if [ ! -f "$fmri_path" ]; then
      echo "⚠️  Missing fMRI file: $fmri_path"
    elif [ ! -f "$label_path" ]; then
      echo "⚠️  Missing label file: $label_path"
    else
      python "$SCRIPT_PATH" \
        --fmri_path     "$fmri_path" \
        --label_path    "$label_path" \
        --graph_label   "$graph_label" \
        --threshold     "$THRESHOLD" \
        --corr_threshold "$CORR_THRESHOLD" \
        --max_distance  "$MAX_DISTANCE" \
        ${ROI_IDS:+--roi_ids "$ROI_IDS"} \
        --out_path      "$out_path"
      echo "✔ Saved graph to $out_path"
    fi

    echo "----- ${subj_id} END   $(date '+%Y-%m-%d %H:%M:%S') -----"
  } > "$tmp" 2>&1

  cat "$tmp" >> "$LOGFILE"
  rm "$tmp"
}

export -f process_subject
export INPUT_DIR OUTPUT_DIR SCRIPT_PATH LOGFILE THRESHOLD CORR_THRESHOLD MAX_DISTANCE ROI_IDS

find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d \
  | parallel -j "$JOBS" process_subject {}

echo "All done — see log at ${LOGFILE}"

