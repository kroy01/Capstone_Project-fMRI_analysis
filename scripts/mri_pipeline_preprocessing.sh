#!/usr/bin/env bash
#
# Usage:
#   mri_pipeline.sh -i <input_nii> -o <output_dir>
#
# Example:
#   mri_pipeline.sh -i structural_input.nii.gz -o output/
#
# Note:
#   This script is for single-volume (3D) MRI data in NIfTI format
#   and will:
#     1) Brain-extract the data
#     2) Perform bias field correction
#     3) Normalize to MNI space
#
# Requirements:
#   - FSL must be installed and $FSLDIR set
#   - The input image is a single-volume .nii or .nii.gz file
#

set -e  # Exit immediately if a command exits with a non-zero status

###############################################################################
# 1) Parse Command-Line Arguments
###############################################################################
INPUT_NII=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -i|--input)
      INPUT_NII="$2"
      shift
      shift
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 -i <input_nii> -o <output_dir>"
      exit 1
      ;;
  esac
done

# Check required arguments
if [[ -z "$INPUT_NII" || -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: You must specify both -i <input_nii> and -o <output_dir>."
  echo "Usage: $0 -i <input_nii> -o <output_dir>"
  exit 1
fi

###############################################################################
# 2) Create Output Subdirectories
###############################################################################
mkdir -p "$OUTPUT_DIR/01_input"
mkdir -p "$OUTPUT_DIR/02_brain_extraction"
mkdir -p "$OUTPUT_DIR/03_bias_correction"
mkdir -p "$OUTPUT_DIR/04_coregistration"
mkdir -p "$OUTPUT_DIR/05_normalization"

###############################################################################
# 3) Copy/Input Handling
###############################################################################
# Copy or link the input NIfTI into the pipeline folder for traceability
cp "$INPUT_NII" "$OUTPUT_DIR/01_input/input_image.nii.gz"

# Define an internal variable to track the "current" working volume
CURRENT_IMAGE="$OUTPUT_DIR/01_input/input_image.nii.gz"

###############################################################################
# 4) Brain Extraction (BET)
###############################################################################
echo "======================================================================"
echo "STEP 1) Brain Extraction (BET)"
echo "======================================================================"
bet "$CURRENT_IMAGE" \
    "$OUTPUT_DIR/02_brain_extraction/brain.nii.gz" \
    -f 0.3 -g 0 -m

# Update CURRENT_IMAGE to the brain-extracted file
CURRENT_IMAGE="$OUTPUT_DIR/02_brain_extraction/brain.nii.gz"

###############################################################################
# 5) Bias Field Correction (FAST)
###############################################################################
echo "======================================================================"
echo "STEP 2) Bias Field Correction (FAST)"
echo "======================================================================"
# FAST typically outputs several files: _seg*, _pve*, etc.
# The bias-corrected image is often named <prefix>_restore.nii.gz
fast -B \
     -o "$OUTPUT_DIR/03_bias_correction/brain_fast" \
     "$CURRENT_IMAGE"

# Now 'brain_fast_restore.nii.gz' should be created properly
if [[ -f "$OUTPUT_DIR/03_bias_correction/brain_fast_restore.nii.gz" ]]; then
  CURRENT_IMAGE="$OUTPUT_DIR/03_bias_correction/brain_fast_restore.nii.gz"
else
  echo "WARNING: FAST bias-corrected image not found. Continuing with original."
fi


###############################################################################
# 6) Coregistration to MNI (FLIRT)
###############################################################################
echo "======================================================================"
echo "STEP 3) Linear Registration (FLIRT) to MNI"
echo "======================================================================"
flirt -in "$CURRENT_IMAGE" \
      -ref "$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz" \
      -out "$OUTPUT_DIR/04_coregistration/brain_flirt.nii.gz" \
      -omat "$OUTPUT_DIR/04_coregistration/affine.mat" \
      -dof 12 \
      -cost corratio

CURRENT_IMAGE="$OUTPUT_DIR/04_coregistration/brain_flirt.nii.gz"

###############################################################################
# 7) Normalization to MNI (FNIRT)
###############################################################################
echo "======================================================================"
echo "STEP 4) Nonlinear Normalization (FNIRT)"
echo "======================================================================"
fnirt --in="$OUTPUT_DIR/03_bias_correction/brain_fast_restore.nii.gz" \
      --aff="$OUTPUT_DIR/04_coregistration/affine.mat" \
      --ref="$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz" \
      --iout="$OUTPUT_DIR/05_normalization/brain_mni.nii.gz" \
      --warpres=6,6,6

# The final normalized image:
FINAL_IMAGE="$OUTPUT_DIR/05_normalization/brain_mni.nii.gz"

echo "======================================================================"
echo "✅ MRI Preprocessing Completed Successfully"
echo "======================================================================"
echo "Final normalized image is at: $FINAL_IMAGE"
exit 0
