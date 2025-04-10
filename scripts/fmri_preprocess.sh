#!/usr/bin/env bash

# Set the path for the log file
LOG_FILE="/home/kroy/Developement/capstone-project/phase_2/fmri_preprocess.log"
touch "$LOG_FILE"

# Redirect all output (stdout and stderr) to include a timestamp and append to the log file
exec > >(while IFS= read -r line; do printf '%s - %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done | tee -a "$LOG_FILE") 2>&1

log() {
    # A simple logging function if you want to explicitly log some messages
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$timestamp - $*"
}

log "Starting fmri preprocessing..."

BASE_DIR="/home/kroy/Developement/capstone-project/phase_2/RESOURCE_FORMATTED"
TEMPLATE_FSF="/home/kroy/Developement/capstone-project/phase_2/src/template.fsf"

###############################
# Pre-stats parameters
SMOOTH_FWHM=5      # e.g., 5 mm
HP_CUTOFF=100      # e.g., 100 s
LP_CUTOFF=-1       # set to -1 if no lowpass
###############################
# Registration parameters
STANDARD_IMAGE_PATH="${FSLDIR}/data/standard/MNI152_T1_2mm_brain"
log "STANDARD_IMAGE_PATH: ${STANDARD_IMAGE_PATH}"

###############################
# Build SUBJECTS array
###############################
SUBJECTS=()

if [ -d "$BASE_DIR" ]; then
    for dir in "$BASE_DIR"/*; do
        if [ -d "$dir" ]; then
            dir_name=$(basename "$dir")
            SUBJECTS+=("$dir_name")
        fi
    done
else
    log "Error: Base directory $BASE_DIR does not exist"
    exit 1
fi

log "SUBJECTS found: ${SUBJECTS[@]}"

###############################
# Loop over subjects
###############################
for SUBJ in "${SUBJECTS[@]}"; do

    SUBJECT_DIR="${BASE_DIR}/${SUBJ}"

    # Original T1 path:
    T1_PATH="${SUBJECT_DIR}/T1.nii.gz"

    # Intermediate outputs for T1:
    T1_REORIENT_PATH="${SUBJECT_DIR}/T1_reorient.nii.gz"
    T1_CROP_PATH="${SUBJECT_DIR}/T1_crop.nii.gz"

    # Final skull-stripped output:
    T1_BRAIN_PATH="${SUBJECT_DIR}/T1_crop_brain.nii.gz"

    # Run the T1 checks/preprocessing:
    if [ -f "$T1_PATH" ]; then
        log "---------------------------------------------------------"
        log "Processing T1 for subject: $SUBJ"
        log "  T1_PATH: $T1_PATH"

        # 1) Reorient T1 to standard
        log "  -> Reorienting T1 to standard..."
        fslreorient2std "$T1_PATH" "$T1_REORIENT_PATH"

        # 2) Crop using robustfov
        log "  -> Cropping field of view with robustfov..."
        robustfov -i "$T1_REORIENT_PATH" -r "$T1_CROP_PATH"

        # 3) Run BET on cropped T1 with robust center estimation (-R) and bias field correction (-B)
        log "  -> Running BET on cropped T1..."
        bet "$T1_CROP_PATH" "$T1_BRAIN_PATH" -R -f 0.5 -g 0 -m -B

    else
        log "Error: T1 file $T1_PATH not found -- skipping subject $SUBJ."
        continue
    fi

    FUNC_PATH="${SUBJECT_DIR}/func.nii.gz"
    OUTPUT_DIR="${SUBJECT_DIR}/prestats_${SUBJ}.feat"
    FSF_FILE="${SUBJECT_DIR}/run_${SUBJ}.fsf"

    log "Processing subject $SUBJ..."
    log "  FUNC_PATH: $FUNC_PATH"

    ############################################
    # 1) Gather Basic Info (TR, NPTS, Dimensions)
    ############################################
    if [ ! -f "$FUNC_PATH" ]; then
        log "Error: functional file $FUNC_PATH does not exist. Skipping subject $SUBJ."
        continue
    fi

    TR=$(fslval "$FUNC_PATH" pixdim4)
    NPTS=$(fslval "$FUNC_PATH" dim4)
    dim1=$(fslval "$FUNC_PATH" dim1)
    dim2=$(fslval "$FUNC_PATH" dim2)
    dim3=$(fslval "$FUNC_PATH" dim3)
    dim4=$(fslval "$FUNC_PATH" dim4)

    # totalVoxels = dim1*dim2*dim3*dim4
    TOTAL_VOXELS=$(( dim1 * dim2 * dim3 * dim4 ))

    log "TR: $TR"
    log "NPTS: $NPTS"
    log "Full 4D dims: ${dim1}x${dim2}x${dim3}x${dim4}"
    log "=> totalVoxels: $TOTAL_VOXELS"

    ############################################
    # 2) Convert smoothing FWHM -> sigma
    ############################################
    SIGMA_SMOOTH=$(python -c "print(${SMOOTH_FWHM}/2.355)")

    ############################################
    # 3) High-pass filter sigma in volumes
    ############################################
    if [[ $HP_CUTOFF == -1 ]]; then
        HP_SIGMA_VOL=-1
    else
        HP_SIGMA_SEC=$(python -c "print(${HP_CUTOFF}/2.0)")
        HP_SIGMA_VOL=$(python -c "print(${HP_SIGMA_SEC}/${TR})")
    fi

    ############################################
    # 4) Low-pass filter sigma in volumes
    ############################################
    if [[ $LP_CUTOFF == -1 ]]; then
        LP_SIGMA_VOL=-1
    else
        LP_SIGMA_SEC=$(python -c "print(${LP_CUTOFF}/2.0)")
        LP_SIGMA_VOL=$(python -c "print(${LP_SIGMA_SEC}/${TR})")
    fi

    ############################################
    # 5) Use estnoise to get Noise (%) & AR(1)
    ############################################
    NOISE_OUT=$(${FSLDIR}/bin/estnoise "$FUNC_PATH" "$SIGMA_SMOOTH" "$HP_SIGMA_VOL" "$LP_SIGMA_VOL" 2>/dev/null)
    NOISE_PCT=$(echo "$NOISE_OUT" | awk '{print $1}')
    AR1=$(echo "$NOISE_OUT"      | awk '{print $2}')

    log "Noise level (%): $NOISE_PCT"
    log "Temporal smoothness (AR1): $AR1"

    ############################################
    # 6) Fill in the FSF placeholders
    ############################################
    sed -e "s|PLACEHOLDER_NOISE|${NOISE_PCT}|g" \
        -e "s|PLACEHOLDER_SMOOTH|${AR1}|g" \
        -e "s|PLACEHOLDER_TR|${TR}|g" \
        -e "s|PLACEHOLDER_NPTS|${NPTS}|g" \
        -e "s|PLACEHOLDER_TOTAL_VOXELS|${TOTAL_VOXELS}|g" \
        -e "s|PLACEHOLDER_FUNC_PATH|${FUNC_PATH}|g" \
        -e "s|PLACEHOLDER_T1_BRAIN_PATH|${T1_BRAIN_PATH}|g" \
        -e "s|PLACEHOLDER_STANDARD_IMAGE_PATH|${STANDARD_IMAGE_PATH}|g" \
        -e "s|PLACEHOLDER_OUTPUT_DIR|${OUTPUT_DIR}|g" \
        "$TEMPLATE_FSF" > "$FSF_FILE"

    ############################################
    # 7) Run FEAT
    ############################################
    log "  -> Running FEAT for subject $SUBJ..."
    feat "$FSF_FILE"

    log "Finished processing $SUBJ. Output: ${OUTPUT_DIR}/filtered_func_data.nii.gz"
    log "---------------------------------------------------------"

done

log "fmri preprocessing completed."
