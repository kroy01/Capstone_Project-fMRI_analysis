######################################
# Basic script-level settings
######################################
LOG_FILE="/export/kroy/Capstone_Project-fMRI_analysis/fmri_preprocess.log"
touch "$LOG_FILE"

BASE_DIR="/export/kroy/Capstone_Project-fMRI_analysis/MCI"
TEMPLATE_FSF="/export/kroy/Capstone_Project-fMRI_analysis/scripts/template.fsf"

SMOOTH_FWHM=5
HP_CUTOFF=100
LP_CUTOFF=-1

STANDARD_IMAGE_PATH="${FSLDIR}/data/standard/MNI152_T1_2mm_brain"

MAX_JOBS="${1:-4}"   # Default parallel jobs = 4

######################################
# Logging helpers
######################################
main_log() {
  # Minimal logging at the script level
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$timestamp - [Main] $*" >> "$LOG_FILE"
}

######################################
# Announce some high-level events
######################################
main_log "================================================================"
main_log "Starting parallel fmri_preprocessing.sh"
main_log "Max parallel jobs = $MAX_JOBS"
main_log "Base directory = $BASE_DIR"
main_log "Standard image path = $STANDARD_IMAGE_PATH"
main_log "================================================================"

######################################
# Build SUBJECTS array
######################################
SUBJECTS=()
if [ -d "$BASE_DIR" ]; then
    for dir in "$BASE_DIR"/*; do
        if [ -d "$dir" ]; then
            dir_name=$(basename "$dir")
            SUBJECTS+=("$dir_name")
        fi
    done
else
    main_log "Error: Base directory $BASE_DIR does not exist"
    exit 1
fi

main_log "Found subjects: ${SUBJECTS[*]}"

######################################
# Function: process_subject()
# - Processes a single subject, capturing
#   all logs into a temporary file, then
#   appending them as one chunk to $LOG_FILE.
######################################
process_subject() {
    local SUBJ="$1"
    local SUBJECT_DIR="${BASE_DIR}/${SUBJ}"

    # Create a temp file for subject's logs
    local TMP_LOG
    TMP_LOG=$(mktemp "/tmp/${SUBJ}_log.XXXXXX")

    # A helper to log for this subject only
    subject_log() {
        local timestamp
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$timestamp - [Subject ${SUBJ}] $*" >> "$TMP_LOG"
    }

    subject_log "========================================================="
    subject_log "Begin processing subject: $SUBJ"

    # Paths
    local T1_PATH="${SUBJECT_DIR}/T1.nii.gz"
    local T1_REORIENT_PATH="${SUBJECT_DIR}/T1_reorient.nii.gz"
    local T1_CROP_PATH="${SUBJECT_DIR}/T1_crop.nii.gz"
    local T1_BRAIN_PATH="${SUBJECT_DIR}/T1_crop_brain.nii.gz"
    local FUNC_PATH="${SUBJECT_DIR}/func.nii.gz"
    local OUTPUT_DIR="${SUBJECT_DIR}/prestats_${SUBJ}.feat"
    local FSF_FILE="${SUBJECT_DIR}/run_${SUBJ}.fsf"

    ######################################
    # Check T1 file
    ######################################
    if [ -f "$T1_PATH" ]; then
        subject_log "T1 file found: $T1_PATH"
        subject_log "-> Reorienting T1 to standard..."
        fslreorient2std "$T1_PATH" "$T1_REORIENT_PATH" 2>&1 | while read -r line; do subject_log "fslreorient2std: $line"; done

        subject_log "-> Cropping with robustfov..."
        robustfov -i "$T1_REORIENT_PATH" -r "$T1_CROP_PATH" 2>&1 | while read -r line; do subject_log "robustfov: $line"; done

        subject_log "-> BET on cropped T1..."
        bet "$T1_CROP_PATH" "$T1_BRAIN_PATH" -R -f 0.5 -g 0 -m -B 2>&1 | while read -r line; do subject_log "bet: $line"; done
    else
        subject_log "ERROR: T1 file $T1_PATH not found. Skipping subject."
        finalize_logs "$SUBJ" "$TMP_LOG"
        return
    fi

    ######################################
    # Check functional file
    ######################################
    if [ ! -f "$FUNC_PATH" ]; then
        subject_log "ERROR: functional file $FUNC_PATH not found. Skipping subject."
        finalize_logs "$SUBJ" "$TMP_LOG"
        return
    fi

    subject_log "Functional file: $FUNC_PATH"

    ############################################
    # 1) Gather Basic Info (TR, NPTS, Dimensions)
    ############################################
    local TR
    local NPTS
    local dim1
    local dim2
    local dim3
    local dim4

    TR=$(fslval "$FUNC_PATH" pixdim4)
    NPTS=$(fslval "$FUNC_PATH" dim4)
    dim1=$(fslval "$FUNC_PATH" dim1)
    dim2=$(fslval "$FUNC_PATH" dim2)
    dim3=$(fslval "$FUNC_PATH" dim3)
    dim4=$(fslval "$FUNC_PATH" dim4)

    local TOTAL_VOXELS=$(( dim1 * dim2 * dim3 * dim4 ))

    subject_log "TR: $TR"
    subject_log "NPTS: $NPTS"
    subject_log "Dims: ${dim1}x${dim2}x${dim3}x${dim4} => total voxels: $TOTAL_VOXELS"

    ############################################
    # 2) Convert smoothing FWHM -> sigma
    ############################################
    local SIGMA_SMOOTH
    SIGMA_SMOOTH=$(python -c "print(${SMOOTH_FWHM}/2.355)")

    ############################################
    # 3) High-pass filter sigma in volumes
    ############################################
    local HP_SIGMA_VOL
    if [[ $HP_CUTOFF == -1 ]]; then
        HP_SIGMA_VOL=-1
    else
        local HP_SIGMA_SEC
        HP_SIGMA_SEC=$(python -c "print(${HP_CUTOFF}/2.0)")
        HP_SIGMA_VOL=$(python -c "print(${HP_SIGMA_SEC}/${TR})")
    fi

    ############################################
    # 4) Low-pass filter sigma in volumes
    ############################################
    local LP_SIGMA_VOL
    if [[ $LP_CUTOFF == -1 ]]; then
        LP_SIGMA_VOL=-1
    else
        local LP_SIGMA_SEC
        LP_SIGMA_SEC=$(python -c "print(${LP_CUTOFF}/2.0)")
        LP_SIGMA_VOL=$(python -c "print(${LP_SIGMA_SEC}/${TR})")
    fi

    ############################################
    # 5) Use estnoise to get Noise (%) & AR(1)
    ############################################
    local NOISE_OUT
    local NOISE_PCT
    local AR1

    NOISE_OUT=$(${FSLDIR}/bin/estnoise "$FUNC_PATH" "$SIGMA_SMOOTH" "$HP_SIGMA_VOL" "$LP_SIGMA_VOL" 2>/dev/null)
    NOISE_PCT=$(echo "$NOISE_OUT" | awk '{print $1}')
    AR1=$(echo "$NOISE_OUT"      | awk '{print $2}')

    subject_log "Noise level (%): $NOISE_PCT"
    subject_log "Temporal smoothness (AR1): $AR1"

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

    subject_log "FSF file created at: $FSF_FILE"

    ############################################
    # 7) Run FEAT
    ############################################
    subject_log "Running FEAT..."
    feat "$FSF_FILE" 2>&1 | while read -r line; do subject_log "feat: $line"; done

    subject_log "Finished processing $SUBJ. Output: ${OUTPUT_DIR}/filtered_func_data.nii.gz"
    subject_log "========================================================="

    finalize_logs "$SUBJ" "$TMP_LOG"
}

######################################
# finalize_logs()
# - Atomically append the subject's entire
#   temporary log file to the main log,
#   then remove the temp file.
######################################
finalize_logs() {
    local SUBJ="$1"
    local TMP_LOG="$2"

    # Write everything in one big chunk:
    {
      echo "================ START of $SUBJ logs ================"
      cat "$TMP_LOG"
      echo "================ END of $SUBJ logs =================="
    } >> "$LOG_FILE"

    rm -f "$TMP_LOG"
}

######################################
# Main loop: spawn parallel processes
######################################
main_log "Launching parallel jobs..."

jobcount=0

for SUBJ in "${SUBJECTS[@]}"; do
    process_subject "$SUBJ" &
    ((jobcount++))

    # If we're at the max, wait for at least one to finish
    if [ "$jobcount" -ge "$MAX_JOBS" ]; then
        # wait -n (in Bash 4.3+) waits until *one* background job finishes
        wait -n
        ((jobcount--))
    fi
done

# Wait for all remaining jobs
wait

main_log "All parallel jobs completed."
main_log "Script finished."
