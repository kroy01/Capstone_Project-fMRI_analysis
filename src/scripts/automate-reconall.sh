#!/usr/bin/env bash
#
# run_reconall.sh
#
# Usage:
#   ./run_reconall.sh <filtered_resource_directory>
#
# Description:
#   - Finds all "*-t1mri.nii.gz" files in <filtered_resource_directory> (including subdirs).
#   - Runs recon-all (with up to 4 parallel jobs) on each T1 file.
#   - Output is stored in the same directory as each T1 file (due to the -sd {//} argument).
#   - The subject name is derived from the file name by removing the ".nii.gz" extension
#     and stripping out "-t1mri".
#
#     Example: "002_S_0295_2011-06-02_CN-t1mri.nii.gz" becomes subject name "002_S_0295_2011-06-02_CN".
#
#   - "-all" runs the full FreeSurfer pipeline.
#   - "-parallel" enables multi-threading within each recon-all job.
#   - The -qcache flag is omitted to speed up the process if you only want labeled masks.

set -euo pipefail

# 1. Check arguments
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <filtered_resource_directory>"
  exit 1
fi

FILTERED_DIR="$1"

if [[ ! -d "${FILTERED_DIR}" ]]; then
  echo "[ERROR] Directory does not exist: ${FILTERED_DIR}"
  exit 1
fi

# 2. Run recon-all in parallel on each T1 file
# Placeholders used by GNU parallel:
#   {}    : full path of the matching file
#   {//}  : parent directory (used with -sd)
#   {/.}  : basename without extension
#   {/.//-t1mri/} : remove the literal "-t1mri" substring from the basename
#
# The -sd {//} argument ensures the output "subject directory" is created
# in the same directory as the file, so recon-all results stay local.

find "${FILTERED_DIR}" -type f -name "*-t1mri.nii.gz" \
  | parallel --jobs 3 \
    'recon-all \
       -sd {//} \
       -s recon_output \
       -i {} \
       -all \
       -parallel'

