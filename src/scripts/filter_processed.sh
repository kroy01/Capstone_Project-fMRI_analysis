#!/usr/bin/env bash
#
# copy_processed_data.sh
#
# Usage:
#   bash copy_processed_data.sh <input_dir> <output_dir>
#
# Description:
#   This script copies processed RS-fMRI (filtered_func_data.nii.gz)
#   and processed T1-MRI (highres2standard.nii.gz) files from FSL's prestats
#   directories into a separate "filtered_resource"-style directory structure.
#
#   For each subject folder within <input_dir>, it looks for:
#       <subjectID>/prestats_<subjectID>.feat/filtered_func_data.nii.gz
#       <subjectID>/prestats_<subjectID>.feat/reg/highres2standard.nii.gz
#
#   Copies them to:
#       <output_dir>/<subjectID>/<subjectID>-rsfmri.nii.gz
#       <output_dir>/<subjectID>/<subjectID>-t1mri.nii.gz
#
#   The script logs each operation and skips subjects if required files are missing.

set -euo pipefail

# -----------------------
# 1. Parse arguments
# -----------------------

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_dir> <output_dir>"
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Create the output directory if it does not exist
mkdir -p "${OUTPUT_DIR}"

# Create a log file with a timestamp
LOGFILE="${OUTPUT_DIR}/copy_log_$(date +'%Y%m%d_%H%M%S').log"

echo "---------------------------------------------------" | tee -a "${LOGFILE}"
echo "Starting copying process at $(date)" | tee -a "${LOGFILE}"
echo "Input directory : ${INPUT_DIR}" | tee -a "${LOGFILE}"
echo "Output directory: ${OUTPUT_DIR}" | tee -a "${LOGFILE}"
echo "---------------------------------------------------" | tee -a "${LOGFILE}"

# -----------------------
# 2. Loop over subjects
# -----------------------

# For each subdirectory in the INPUT_DIR
for subject_dir in "${INPUT_DIR}"/*; do
  # Only proceed if it's a directory
  if [[ -d "${subject_dir}" ]]; then
    subject_id=$(basename "${subject_dir}")

    echo "Processing subject: ${subject_id}" | tee -a "${LOGFILE}"

    # Construct the expected paths for the processed files
    rsfmri_path="${subject_dir}/prestats_${subject_id}.feat/filtered_func_data.nii.gz"
    t1mri_path="${subject_dir}/prestats_${subject_id}.feat/reg/highres2standard.nii.gz"

    # Check existence of both
    if [[ ! -f "${rsfmri_path}" ]]; then
      echo "  [WARNING] Missing RS-fMRI file for ${subject_id}. Skipping..." | tee -a "${LOGFILE}"
      continue
    fi
    if [[ ! -f "${t1mri_path}" ]]; then
      echo "  [WARNING] Missing T1-MRI file for ${subject_id}. Skipping..." | tee -a "${LOGFILE}"
      continue
    fi

    # Make output subject directory
    output_subdir="${OUTPUT_DIR}/${subject_id}"
    mkdir -p "${output_subdir}"

    # Define output filenames
    rsfmri_out="${output_subdir}/${subject_id}-rsfmri.nii.gz"
    t1mri_out="${output_subdir}/${subject_id}-t1mri.nii.gz"

    # Copy files
    cp "${rsfmri_path}" "${rsfmri_out}"
    cp "${t1mri_path}" "${t1mri_out}"

    echo "  Copied RS-fMRI -> ${rsfmri_out}" | tee -a "${LOGFILE}"
    echo "  Copied T1-MRI  -> ${t1mri_out}" | tee -a "${LOGFILE}"
  fi
done

echo "---------------------------------------------------" | tee -a "${LOGFILE}"
echo "Copying process completed at $(date)" | tee -a "${LOGFILE}"
echo "---------------------------------------------------" | tee -a "${LOGFILE}"

