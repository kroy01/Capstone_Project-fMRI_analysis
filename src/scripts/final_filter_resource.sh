#!/usr/bin/env bash
# ------------------------------------------------------------
#  final_filter_resource.sh
#  Copies rsfMRI, T1‑MRI and converts aparc+aseg label maps
#  for every subject from an input directory to an output
#  directory.
#
#  Usage:  ./final_filter_resource.sh  <INPUT_DIR>  <OUTPUT_DIR>
# ------------------------------------------------------------

set -euo pipefail

# -------------- help message --------------------------------
usage() {
  cat <<EOF
Usage: $(basename "$0") <INPUT_DIR> <OUTPUT_DIR>

  <INPUT_DIR>    Root directory that holds one sub‑directory per subject
                 (the "filtered resources" tree produced by recon‑all).

  <OUTPUT_DIR>   Destination directory where the script will place, for
                 each subject:
                   • *_rsfmri.nii.gz
                   • *_t1mri.nii.gz
                   • <subject>_aparc+aseg.nii.gz  (converted from MGZ)

Notes
-----
* Requires FreeSurfer's 'mri_convert' to be on \$PATH.
* The script skips a subject if any of the three expected files
  cannot be located.

EOF
  exit 1
}

# -------------- argument parsing ----------------------------
[[ $# -ne 2 ]] && usage
IN_DIR=$(realpath "$1")
OUT_DIR=$(realpath "$2")

command -v mri_convert >/dev/null 2>&1 || {
  echo "ERROR: 'mri_convert' not found in PATH.  Load your FreeSurfer env first." >&2
  exit 2
}

# -------------- main loop -----------------------------------
echo "Scanning subjects in: $IN_DIR"
for subj_path in "$IN_DIR"/*/; do
  [[ -d $subj_path ]] || continue
  subj=$(basename "$subj_path")
  echo "▶  Processing subject: $subj"

  # Match either hyphen or underscore before the modality tag
  rsf_file=$(ls "$subj_path"/*[-_]rsfmri.nii.gz 2>/dev/null || true)
  t1_file=$(ls "$subj_path"/*[-_]t1mri.nii.gz   2>/dev/null || true)
  mgz_file="$subj_path/recon_output/mri/aparc+aseg.mgz"

  # Check availability
  if [[ -z $rsf_file || -z $t1_file || ! -f $mgz_file ]]; then
    echo "    ⚠  Missing rsfMRI, T1 or aparc+aseg for $subj — skipping." >&2
    continue
  fi

  # Create subject destination
  dest="$OUT_DIR/$subj"
  mkdir -p "$dest"

  # Copy rsfMRI & T1
  cp -p "$rsf_file" "$dest/"
  cp -p "$t1_file"  "$dest/"

  # Convert label map to NIfTI
  mri_convert \
      --out_orientation RAS \
      "$mgz_file" \
      "$dest/${subj}_aparc+aseg.nii.gz" \
      >/dev/null

  echo "    ✓  Exported to $dest"
done

echo "All done."

