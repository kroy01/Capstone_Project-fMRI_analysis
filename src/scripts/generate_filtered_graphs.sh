#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   filter_all_graphs.sh <input_graph_dir> <output_dir> <roi_list> [num_jobs]
#
#   <input_graph_dir>: directory with *_bold_complete.pt files
#   <output_dir>:       where filtered .pt files + log will go
#   <roi_list>:         comma-separated ROI IDs to keep, e.g. "17,18,53,54"
#   [num_jobs]:         optional parallelism (GNU parallel -j); default=1

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <input_graph_dir> <output_dir> <roi_list> [num_jobs]"
  exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
ROI_LIST=$3
JOBS=${4:-1}                                      # default 1
FILTER_SCRIPT="/export/kroy/Capstone_Project-fMRI_analysis/gnn/src/filter_fmri_graph.py"     # ← adjust to your filter script

mkdir -p "$OUTPUT_DIR"
LOGFILE="${OUTPUT_DIR}/graph_filter.log"
: > "$LOGFILE"   # truncate or create

process_file() {
  input_path="$1"
  name="$(basename "$input_path" _complete.pt)"  
  output_path="${OUTPUT_DIR}/${name}_filtered.pt"

  tmp=$(mktemp)
  {
    echo "----- ${base} START $(date '+%Y-%m-%d %H:%M:%S') -----"
    echo "Keeping ROIs: ${ROI_LIST}"
    if [ ! -f "$input_path" ]; then
      echo "⚠️  Missing input graph: $input_path"
    else
      python3 "$FILTER_SCRIPT" \
        --input "$input_path" \
        --rois "$ROI_LIST" \
        --output "$output_path"
      echo "✔ Saved filtered graph: $output_path"
    fi
    echo "----- ${base} END   $(date '+%Y-%m-%d %H:%M:%S') -----"
  } > "$tmp" 2>&1

  cat "$tmp" >> "$LOGFILE"
  rm "$tmp"
}

export -f process_file
export OUTPUT_DIR ROI_LIST FILTER_SCRIPT LOGFILE

find "$INPUT_DIR" -maxdepth 1 -type f -name "*_bold_complete.pt" \
  | parallel -j "$JOBS" process_file {}

echo "All done — see log at ${LOGFILE}"

