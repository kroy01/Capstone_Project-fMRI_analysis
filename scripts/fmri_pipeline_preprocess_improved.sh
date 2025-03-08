#!/usr/bin/env bash

###############################################################################
# Example: 
#   ./fmri_preprocess.sh -i /path/to/dicom_dir -o /path/to/output_dir
#
# This script performs:
#   1) DICOM to NIfTI conversion (4D)
#   2) Slice timing correction
#   3) Motion correction (MCFLIRT)
#   4) Mean image extraction
#   5) Brain extraction (BET)
#   6) Masking the 4D data with BET mask
#   7) Intensity scaling
#   8) Linear registration (FLIRT) to MNI
#   9) Non-linear normalization (FNIRT) to MNI
#
# Final preprocessed 4D data (in MNI space) is found in:
#   $OUTPUT_DIR/09_normalization/normalized_4D.nii.gz
###############################################################################

set -e  # Exit if any command fails

###############################################################################
# 1) Parse Command-Line Arguments
###############################################################################
INPUT_DIR=""
OUTPUT_DIR=""

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
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 -i <input_dir> -o <output_dir>"
      exit 1
      ;;
  esac
done

# Check for required arguments
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: You must specify both -i <input_dir> and -o <output_dir>."
  echo "Usage: $0 -i <input_dir> -o <output_dir>"
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
# 3) DICOM to NIfTI Conversion
###############################################################################
echo "======================================================================"
echo "1) DICOM to NIfTI Conversion"
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
# 10) Linear Registration to MNI (FLIRT)
###############################################################################
echo "======================================================================"
echo "8) Linear Registration (FLIRT) to MNI"
echo "======================================================================"
# Create a mean image of the scaled 4D to drive the linear registration
fslmaths "$SCALED_4D" -Tmean "$OUTPUT_DIR/08_coregistration/intensity_scaled_mean.nii.gz"

SCALED_MEAN_3D="$OUTPUT_DIR/08_coregistration/intensity_scaled_mean.nii.gz"

# Register the mean functional to the MNI template
flirt -in "$SCALED_MEAN_3D" \
      -ref "$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz" \
      -out "$OUTPUT_DIR/08_coregistration/coregistered_mean.nii.gz" \
      -omat "$OUTPUT_DIR/08_coregistration/coregistered_mean.mat" \
      -dof 12 -cost corratio

###############################################################################
# 11) Non-Linear Normalization (FNIRT)
###############################################################################
echo "======================================================================"
echo "9) Non-Linear Normalization (FNIRT)"
echo "======================================================================"
fnirt --in="$SCALED_MEAN_3D" \
      --aff="$OUTPUT_DIR/08_coregistration/coregistered_mean.mat" \
      --ref="$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz" \
      --iout="$OUTPUT_DIR/09_normalization/fmri_mean_normalized.nii.gz" \
      --cout="$OUTPUT_DIR/09_normalization/fmri_mean_warpcoef.nii.gz" \
      --warpres=6,6,6

# Now apply the warp (computed above) to the entire 4D dataset
applywarp --ref="$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz" \
          --in="$SCALED_4D" \
          --out="$OUTPUT_DIR/09_normalization/normalized_4D.nii.gz" \
          --warp="$OUTPUT_DIR/09_normalization/fmri_mean_warpcoef.nii.gz"

echo ""
echo "======================================================================"
echo "✅ fMRI preprocessing complete! Final 4D dataset in MNI space:"
echo "   $OUTPUT_DIR/09_normalization/normalized_4D.nii.gz"
echo "======================================================================"
