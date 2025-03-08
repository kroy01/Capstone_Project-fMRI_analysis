#!/usr/bin/env bash

###############################################################################
# Example:
#   ./fmri_pipeline_preprocess_improved.sh -i /path/to/dicom_dir -o /path/to/output
#
# This script performs:
#   1) DICOM to NIfTI (4D) conversion
#   2) Slice timing correction
#   3) Motion correction (MCFLIRT)
#   4) Mean image extraction
#   5) Brain extraction (BET)
#   6) Mask the 4D data
#   7) Intensity scaling
#   8) Linear registration (FLIRT) - for both 3D mean and entire 4D
#   9) Non-linear normalization (FNIRT) - optional step, then apply to 4D
#
# Final 4D datasets:
#   * Affine-registered 4D => $OUTPUT_DIR/08_coregistration/4d_coregistered_affine.nii.gz
#   * Nonlinearly normalized 4D => $OUTPUT_DIR/09_normalization/4d_normalized_fnirt.nii.gz
###############################################################################

set -e  # Exit if any command fails

###############################################################################
# 1) Parse Command-Line Arguments
###############################################################################
INPUT_DIR=""
OUTPUT_DIR=""
DO_NONLINEAR=true  # Toggle if you do or don't want to do fnirt

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -i|--input)
      INPUT_DIR="$2"
      shift
      shift
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --no-fnirt)
      DO_NONLINEAR=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 -i <input_dir> -o <output_dir> [--no-fnirt]"
      exit 1
      ;;
  esac
done

# Check for required arguments
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: You must specify both -i <input_dir> and -o <output_dir>."
  echo "Usage: $0 -i <input_dir> -o <output_dir> [--no-fnirt]"
  exit 1
fi

###############################################################################
# 2) Create Output Subdirectories
###############################################################################
mkdir -p "$OUTPUT_DIR/01_dicom_conversion"
mkdir -p "$OUTPUT_DIR/02_slice_timing"
mkdir -p "$OUTPUT_DIR/03_motion_correction"
mkdir -p "$OUTPUT_DIR/04_mean_image"
mkdir -p "$OUTPUT_DIR/05_brain_extraction"
mkdir -p "$OUTPUT_DIR/06_masked_4d"
mkdir -p "$OUTPUT_DIR/07_intensity_scaling"
mkdir -p "$OUTPUT_DIR/08_coregistration"
mkdir -p "$OUTPUT_DIR/09_normalization"

###############################################################################
# 3) DICOM to NIfTI (4D) Conversion
###############################################################################
echo "======================================================================"
echo "1) DICOM to NIfTI Conversion (4D)"
echo "======================================================================"
dcm2niix -o "$OUTPUT_DIR/01_dicom_conversion" \
         -f "fmri_4d" \
         "$INPUT_DIR"

# Expect output: fmri_4d.nii.gz
FMRI_4D="$OUTPUT_DIR/01_dicom_conversion/fmri_4d.nii.gz"

###############################################################################
# 4) Slice Timing Correction
###############################################################################
echo "======================================================================"
echo "2) Slice Timing Correction"
echo "======================================================================"
slicetimer -i "$FMRI_4D" \
           -o "$OUTPUT_DIR/02_slice_timing/slice_timing_corrected.nii.gz" \
           --odd

STC_4D="$OUTPUT_DIR/02_slice_timing/slice_timing_corrected.nii.gz"

###############################################################################
# 5) Motion Correction (MCFLIRT)
###############################################################################
echo "======================================================================"
echo "3) Motion Correction"
echo "======================================================================"
mcflirt -in "$STC_4D" \
        -out "$OUTPUT_DIR/03_motion_correction/motion_corrected_4d.nii.gz" \
        -plots -cost mutualinfo

MC_4D="$OUTPUT_DIR/03_motion_correction/motion_corrected_4d.nii.gz"

###############################################################################
# 6) Mean Image Extraction (Tmean)
###############################################################################
echo "======================================================================"
echo "4) Mean Image Extraction"
echo "======================================================================"
fslmaths "$MC_4D" \
         -Tmean "$OUTPUT_DIR/04_mean_image/fmri_mean.nii.gz"

MEAN_3D="$OUTPUT_DIR/04_mean_image/fmri_mean.nii.gz"

###############################################################################
# 7) Brain Extraction (BET) on Mean Image
###############################################################################
echo "======================================================================"
echo "5) Brain Extraction (BET)"
echo "======================================================================"
bet "$MEAN_3D" \
    "$OUTPUT_DIR/05_brain_extraction/fmri_mean_brain.nii.gz" \
    -f 0.4 -g 0 -m

BET_3D="$OUTPUT_DIR/05_brain_extraction/fmri_mean_brain.nii.gz"
BET_MASK="$OUTPUT_DIR/05_brain_extraction/fmri_mean_brain_mask.nii.gz"

###############################################################################
# 8) Mask the 4D Data
###############################################################################
echo "======================================================================"
echo "6) Apply Brain Mask to 4D"
echo "======================================================================"
fslmaths "$MC_4D" \
         -mas "$BET_MASK" \
         "$OUTPUT_DIR/06_masked_4d/fmri_4d_masked.nii.gz"

MASKED_4D="$OUTPUT_DIR/06_masked_4d/fmri_4d_masked.nii.gz"

###############################################################################
# 9) Intensity Scaling
###############################################################################
echo "======================================================================"
echo "7) Intensity Scaling"
echo "======================================================================"
fslmaths "$MASKED_4D" \
         -ing 10000 \
         "$OUTPUT_DIR/07_intensity_scaling/intensity_scaled_4d.nii.gz"

SCALED_4D="$OUTPUT_DIR/07_intensity_scaling/intensity_scaled_4d.nii.gz"

###############################################################################
# 10) Linear Registration (FLIRT) - 3D & 4D
###############################################################################
echo "======================================================================"
echo "8) Linear Registration (FLIRT)"
echo "======================================================================"
# Create a mean image of the scaled 4D to drive the linear registration
fslmaths "$SCALED_4D" -Tmean "$OUTPUT_DIR/08_coregistration/intensity_scaled_mean.nii.gz"
SCALED_MEAN_3D="$OUTPUT_DIR/08_coregistration/intensity_scaled_mean.nii.gz"

# Register the mean functional to the MNI template (affine)
flirt -in "$SCALED_MEAN_3D" \
      -ref "$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz" \
      -out "$OUTPUT_DIR/08_coregistration/coregistered_mean_affine.nii.gz" \
      -omat "$OUTPUT_DIR/08_coregistration/coregistered_mean_affine.mat" \
      -dof 12 -cost corratio

# Now apply that same transform to the entire 4D dataset
flirt -in "$SCALED_4D" \
      -ref "$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz" \
      -applyxfm -init "$OUTPUT_DIR/08_coregistration/coregistered_mean_affine.mat" \
      -out "$OUTPUT_DIR/08_coregistration/4d_coregistered_affine.nii.gz"

AFFINE_4D="$OUTPUT_DIR/08_coregistration/4d_coregistered_affine.nii.gz"

###############################################################################
# 11) (Optional) Non-linear Normalization (FNIRT)
###############################################################################
if [ "$DO_NONLINEAR" = true ]; then
  echo "======================================================================"
  echo "9) Non-linear Normalization (FNIRT)"
  echo "======================================================================"
  # We'll run FNIRT on the mean 3D volume (already affine-registered).
  # So let's create a new mean from the AFFINE_4D or simply reuse the
  # “coregistered_mean_affine” from the step above.

  # (A) Recreate a mean from the AFFINE_4D (safer, in case the intensity changed):
  fslmaths "$AFFINE_4D" -Tmean "$OUTPUT_DIR/09_normalization/affine_4d_mean.nii.gz"
  AFFINE_MEAN_3D="$OUTPUT_DIR/09_normalization/affine_4d_mean.nii.gz"

  # (B) Now run FNIRT with the affine mat from above
  fnirt --in="$AFFINE_MEAN_3D" \
        --aff="$OUTPUT_DIR/08_coregistration/coregistered_mean_affine.mat" \
        --ref="$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz" \
        --iout="$OUTPUT_DIR/09_normalization/3d_mean_fnirt.nii.gz" \
        --cout="$OUTPUT_DIR/09_normalization/mean_warpcoef.nii.gz" \
        --warpres=6,6,6

  # (C) Apply the warp to the entire 4D
  applywarp --ref="$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz" \
            --in="$AFFINE_4D" \
            --out="$OUTPUT_DIR/09_normalization/4d_normalized_fnirt.nii.gz" \
            --warp="$OUTPUT_DIR/09_normalization/mean_warpcoef.nii.gz"

  echo ""
  echo "======================================================================"
  echo "✅ 4D data now nonlinearly normalized: "
  echo "   $OUTPUT_DIR/09_normalization/4d_normalized_fnirt.nii.gz"
  echo "======================================================================"

else
  echo ""
  echo "======================================================================"
  echo "Nonlinear normalization (FNIRT) is skipped (--no-fnirt)."
  echo "Affine-registered 4D dataset:"
  echo "   $OUTPUT_DIR/08_coregistration/4d_coregistered_affine.nii.gz"
  echo "======================================================================"
fi
