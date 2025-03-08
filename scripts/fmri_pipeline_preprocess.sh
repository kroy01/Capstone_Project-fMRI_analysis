#!/usr/bin/env bash
#
# Usage:
#   fmri_pipeline.sh -i <input_dir> -o <output_dir> [options]
#
# Options:
#   -smoothing        Run spatial smoothing
#   -hp_filtering     Run high-pass filtering
#   -ica_denoising    Run ICA-based denoising
#
# Example:
#   fmri_pipeline.sh -i input/ -o output_v2/ -smoothing -hp_filtering -ica_denoising
#

set -e  # Exit if any command fails

###############################################################################
# 1) Parse Command-Line Arguments
###############################################################################
SMOOTHING=false
HP_FILTERING=false
ICA_DENOISING=false
INPUT_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]
do
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
    -smoothing)
      SMOOTHING=true
      shift
      ;;
    -hp_filtering)
      HP_FILTERING=true
      shift
      ;;
    -ica_denoising)
      ICA_DENOISING=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 -i <input_dir> -o <output_dir> [-smoothing] [-hp_filtering] [-ica_denoising]"
      exit 1
      ;;
  esac
done

# Check required arguments
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: You must specify both -i <input_dir> and -o <output_dir>."
  echo "Usage: $0 -i <input_dir> -o <output_dir> [-smoothing] [-hp_filtering] [-ica_denoising]"
  exit 1
fi

###############################################################################
# 2) Create Output Subdirectories
###############################################################################
mkdir -p "$OUTPUT_DIR/01_dicom_conversion"
mkdir -p "$OUTPUT_DIR/02_slice_timing"
mkdir -p "$OUTPUT_DIR/03_mean_image"
mkdir -p "$OUTPUT_DIR/04_brain_extraction"
mkdir -p "$OUTPUT_DIR/05_intensity_scaling"
mkdir -p "$OUTPUT_DIR/06_bias_correction"
mkdir -p "$OUTPUT_DIR/07_motion_correction"
mkdir -p "$OUTPUT_DIR/08_spatial_smoothing"
mkdir -p "$OUTPUT_DIR/09_coregistration"
mkdir -p "$OUTPUT_DIR/10_normalization"
mkdir -p "$OUTPUT_DIR/11_highpass_filtering"
mkdir -p "$OUTPUT_DIR/12_ica_denoising"

###############################################################################
# 3) Reordered Pipeline Steps
###############################################################################

echo "======================================================================"
echo "1) DICOM to NIfTI Conversion"
echo "======================================================================"
dcm2niix -o "$OUTPUT_DIR/01_dicom_conversion" \
         -f "converted_output" \
         "$INPUT_DIR"

# ---------------------------------------------------------------------------

echo "======================================================================"
echo "2) Slice Timing Correction"
echo "======================================================================"
slicetimer -i "$OUTPUT_DIR/01_dicom_conversion/converted_output.nii.gz" \
           -o "$OUTPUT_DIR/02_slice_timing/slice_timing_corrected.nii.gz" \
           --odd

# ---------------------------------------------------------------------------

echo "======================================================================"
echo "3) Create Mean Image (Tmean)"
echo "======================================================================"
fslmaths "$OUTPUT_DIR/02_slice_timing/slice_timing_corrected.nii.gz" \
         -Tmean "$OUTPUT_DIR/03_mean_image/slice_timing_corrected_mean.nii.gz"

# ---------------------------------------------------------------------------

echo "======================================================================"
echo "4) Brain Extraction (BET)"
echo "======================================================================"
bet "$OUTPUT_DIR/03_mean_image/slice_timing_corrected_mean.nii.gz" \
    "$OUTPUT_DIR/04_brain_extraction/slice_timing_corrected_mean_brain.nii.gz" \
    -f 0.4 -g 0 -m

# ---------------------------------------------------------------------------

echo "======================================================================"
echo "5) Intensity Scaling (after BET)"
echo "======================================================================"
fslmaths "$OUTPUT_DIR/02_slice_timing/slice_timing_corrected.nii.gz" \
         -ing 10000 \
         "$OUTPUT_DIR/05_intensity_scaling/intensity_scaled.nii.gz"

# ---------------------------------------------------------------------------

echo "======================================================================"
echo "6) Bias Field Correction (FAST)"
echo "======================================================================"
# The original script did FAST on the mean_brain file
fast -b "$OUTPUT_DIR/04_brain_extraction/slice_timing_corrected_mean_brain.nii.gz"

# Move FAST output_v2 files into the bias_correction folder
mv "$OUTPUT_DIR/04_brain_extraction/slice_timing_corrected_mean_"* \
   "$OUTPUT_DIR/06_bias_correction/" 2>/dev/null || true

# ---------------------------------------------------------------------------

echo "======================================================================"
echo "7) Motion Correction (MCFLIRT)"
echo "======================================================================"
# The original script does MCFLIRT on the mean brain from the bias-corrected folder
mcflirt -in "$OUTPUT_DIR/06_bias_correction/slice_timing_corrected_mean_brain.nii.gz" \
        -out "$OUTPUT_DIR/07_motion_correction/motion_corrected.nii.gz" \
        -cost mutualinfo -smooth 0.5 -plots

# ---------------------------------------------------------------------------

echo "======================================================================"
echo "8) Spatial Smoothing [optional]"
echo "======================================================================"
if [ "$SMOOTHING" = true ]; then
    echo "Spatial smoothing is ON."
    fslmaths "$OUTPUT_DIR/07_motion_correction/motion_corrected.nii.gz" \
             -s 1.5 \
             "$OUTPUT_DIR/08_spatial_smoothing/smoothed.nii.gz"
else
    echo "Spatial smoothing is OFF. Skipping."
fi

# ---------------------------------------------------------------------------

echo "======================================================================"
echo "9) Coregistration to MNI (FLIRT)"
echo "======================================================================"
# The original script did FLIRT using the intensity-scaled data
flirt -in "$OUTPUT_DIR/05_intensity_scaling/intensity_scaled.nii.gz" \
      -ref "$FSLDIR/data/standard/MNI152_T1_2mm.nii.gz" \
      -out "$OUTPUT_DIR/09_coregistration/coregistered.nii.gz" \
      -dof 12 \
      -cost corratio \
      -omat "$OUTPUT_DIR/09_coregistration/coregistration.mat"

# ---------------------------------------------------------------------------

echo "======================================================================"
echo "10) Normalization to MNI (FNIRT)"
echo "======================================================================"
fnirt --in="$OUTPUT_DIR/05_intensity_scaling/intensity_scaled.nii.gz" \
      --aff="$OUTPUT_DIR/09_coregistration/coregistration.mat" \
      --ref="$FSLDIR/data/standard/MNI152_T1_2mm.nii.gz" \
      --warpres=6,6,6 \
      --iout="$OUTPUT_DIR/10_normalization/normalized_mni.nii.gz"

# Decide what the “final” file to feed into optional steps is
FINAL_IMAGE="$OUTPUT_DIR/10_normalization/normalized_mni.nii.gz"

# ---------------------------------------------------------------------------

echo "======================================================================"
echo "11) High-pass Filtering [optional]"
echo "======================================================================"
if [ "$HP_FILTERING" = true ]; then
    echo "High-pass filtering is ON."
    fslmaths "$FINAL_IMAGE" \
             -bptf 20 -1 \
             -add 10000 \
             -thr 10 \
             "$OUTPUT_DIR/11_highpass_filtering/highpass_filtered.nii.gz"
    FINAL_IMAGE="$OUTPUT_DIR/11_highpass_filtering/highpass_filtered.nii.gz"
else
    echo "High-pass filtering is OFF. Skipping."
fi

# ---------------------------------------------------------------------------

echo "======================================================================"
echo "12) ICA Denoising [optional]"
echo "======================================================================"
if [ "$ICA_DENOISING" = true ]; then
    echo "ICA denoising is ON."
    melodic -i "$FINAL_IMAGE" \
            -o "$OUTPUT_DIR/12_ica_denoising" \
            --nobet -d 30 --bgthreshold=2 --mmthresh=0.5
else
    echo "ICA denoising is OFF. Skipping."
fi

# ---------------------------------------------------------------------------

echo "✅ Full preprocessing pipeline completed with optimized parameters."
