#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   generate_clustered_graphs.sh \
#     --inp_dir           <input_graph_dir> \
#     --out_dir           <output_dir> \
#     [--voxels_per_cluster <n>] \
#     [--min_cluster_count  <n>] \
#     [--max_cluster_count  <n>] \
#     [--corr_threshold     <float>] \
#     [--max_distance       <float>] \
#     [--jobs               <n>]

# defaults
JOBS=1
VOXELS_PER_CLUSTER=100
MIN_CLUSTER_COUNT=1
MAX_CLUSTER_COUNT=50
CORR_THRESHOLD=0.3
MAX_DISTANCE=10.0
CLUSTER_SCRIPT="/export/kroy/Capstone_Project-fMRI_analysis/gnn/src/generate_clustered_graph.py"

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --inp_dir)            INPUT_DIR="$2";              shift 2 ;;
    --out_dir)            OUTPUT_DIR="$2";             shift 2 ;;
    --jobs)               JOBS="$2";                   shift 2 ;;
    --voxels_per_cluster) VOXELS_PER_CLUSTER="$2";     shift 2 ;;
    --min_cluster_count)  MIN_CLUSTER_COUNT="$2";      shift 2 ;;
    --max_cluster_count)  MAX_CLUSTER_COUNT="$2";      shift 2 ;;
    --corr_threshold)     CORR_THRESHOLD="$2";         shift 2 ;;
    --max_distance)       MAX_DISTANCE="$2";           shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# validate required
if [ -z "${INPUT_DIR:-}" ] || [ -z "${OUTPUT_DIR:-}" ]; then
  echo "ERROR: --inp_dir and --out_dir are required."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOGFILE="${OUTPUT_DIR}/graph_clustering.log"
: > "$LOGFILE"

process_file() {
  local inpt="$1"
  local base="$(basename "$inpt" _complete.pt)"
  local outp="${OUTPUT_DIR}/${base}_clustered.pt"

  local tmp
  tmp=$(mktemp)
  {
    echo "----- ${base} START $(date '+%Y-%m-%d %H:%M:%S') -----"
    echo "voxels_per_cluster: ${VOXELS_PER_CLUSTER}"
    echo "min_cluster_count:  ${MIN_CLUSTER_COUNT}"
    echo "max_cluster_count:  ${MAX_CLUSTER_COUNT}"
    echo "corr_threshold:     ${CORR_THRESHOLD}"
    echo "max_distance:       ${MAX_DISTANCE}"
    echo "jobs:               ${JOBS}"
    if [ ! -f "$inpt" ]; then
      echo "⚠️  Missing input graph: $inpt"
    else
      python "$CLUSTER_SCRIPT" \
        --input               "$inpt" \
        --output              "$outp" \
        --voxels_per_cluster  "$VOXELS_PER_CLUSTER" \
        --min_clusters        "$MIN_CLUSTER_COUNT" \
        --max_clusters        "$MAX_CLUSTER_COUNT" \
        --corr_threshold      "$CORR_THRESHOLD" \
        --max_distance        "$MAX_DISTANCE"
      echo "✔ Saved clustered graph:  $outp"
      echo "✔ Saved mapping JSON:     ${outp%.pt}_mapping.json"
    fi
    echo "----- ${base} END   $(date '+%Y-%m-%d %H:%M:%S') -----"
  } > "$tmp" 2>&1

  cat "$tmp" >> "$LOGFILE"
  rm "$tmp"
}

export -f process_file
export INPUT_DIR OUTPUT_DIR VOXELS_PER_CLUSTER MIN_CLUSTER_COUNT MAX_CLUSTER_COUNT CORR_THRESHOLD MAX_DISTANCE CLUSTER_SCRIPT LOGFILE

find "$INPUT_DIR" -maxdepth 1 -type f -name "*_bold_complete.pt" \
  | parallel --will-cite -j "$JOBS" process_file {}

echo "All done — see log at ${LOGFILE}"

