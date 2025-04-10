
#!/usr/bin/env bash
###############################################################################
# Usage: convert_scans.sh <input_parent_dir> <output_dir> <LABEL>
# Example: ./convert_scans.sh /path/to/ADNI_SAMPLE /path/to/RESOURCE_FORMATTED MCI
#
# Requirements: dcm2niix must be installed and available in your PATH.
###############################################################################

# Function to display error messages
error_exit() {
  echo "ERROR: $1" >&2
  echo "Exiting script."
  exit 1
}

# Function to display warning messages
warning_msg() {
  echo "WARNING: $1" >&2
}

# Validate arguments.
if [ "$#" -ne 3 ]; then
  error_exit "Usage: $0 <input_parent_dir> <output_dir> <LABEL>"
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
LABEL="$3"

# Verify input directory exists
if [ ! -d "$INPUT_DIR" ]; then
  error_exit "Input directory does not exist: $INPUT_DIR"
fi

# Verify dcm2niix is available.
if ! command -v dcm2niix >/dev/null 2>&1; then
  error_exit "dcm2niix not found. Please install it or add it to your PATH."
fi

# Create the output directory if it doesn't exist.
if ! mkdir -p "$OUTPUT_DIR"; then
  error_exit "Failed to create output directory: $OUTPUT_DIR"
fi

echo "Processing subjects from: $INPUT_DIR"
echo "Output will be stored in: $OUTPUT_DIR"
echo "Label used: $LABEL"
echo "-------------------------------------------------------"

# Loop over each subject folder.
for SUBJECT_DIR in "$INPUT_DIR"/*; do
  if [ ! -d "$SUBJECT_DIR" ]; then
    warning_msg "Skipping non-directory: $SUBJECT_DIR"
    continue
  fi

  SUB_ID=$(basename "$SUBJECT_DIR")
  echo "Processing subject: $SUB_ID"

  # Loop over each scan type folder inside the subject folder.
  for SCAN_TYPE_DIR in "$SUBJECT_DIR"/*; do
    if [ ! -d "$SCAN_TYPE_DIR" ]; then
      warning_msg "Skipping non-directory in subject $SUB_ID: $SCAN_TYPE_DIR"
      continue
    fi

    # Determine the scan type based on the folder name (case-insensitive).
    SCAN_DIR_NAME=$(basename "$SCAN_TYPE_DIR")
    if [[ "${SCAN_DIR_NAME,,}" == *fmri* ]]; then
      SCAN_TYPE="func"
    else
      SCAN_TYPE="T1"
    fi

    # Loop over each date folder inside the scan type folder.
    for DATE_DIR in "$SCAN_TYPE_DIR"/*; do
      if [ ! -d "$DATE_DIR" ]; then
        warning_msg "Skipping non-directory in scan type $SCAN_TYPE: $DATE_DIR"
        continue
      fi

      FULL_DATE_FOLDER=$(basename "$DATE_DIR")
      # Extract only the date part (first 10 characters, e.g., "2019-03-06")
      DATE_ONLY=${FULL_DATE_FOLDER:0:10}

      # Create (or reuse) the output resource folder.
      RESOURCE_DIR="${OUTPUT_DIR}/${SUB_ID}_${DATE_ONLY}_${LABEL}"
      if ! mkdir -p "$RESOURCE_DIR"; then
        warning_msg "Failed to create resource directory: $RESOURCE_DIR"
        continue
      fi

      echo "  Processing scan from: $DATE_DIR"
      echo "    -> Date: $DATE_ONLY, Scan Type: $SCAN_TYPE"

      # Check if the file for this scan type already exists in the resource folder.
      if [ -f "${RESOURCE_DIR}/${SCAN_TYPE}.nii" ] || [ -f "${RESOURCE_DIR}/${SCAN_TYPE}.nii.gz" ]; then
         echo "    ${SCAN_TYPE} file already exists in ${RESOURCE_DIR}; skipping."
         continue
      fi

      # Search recursively for an existing NIfTI file in the date folder.
      NII_FILE=$(find "$DATE_DIR" -type f \( -iname "*.nii" -o -iname "*.nii.gz" \) | head -n 1)
      if [ -n "$NII_FILE" ]; then
         echo "    Found NIfTI file: $NII_FILE. Attempting to copy to resource folder."
         # Preserve the extension (either .nii or .nii.gz)
         if [[ "$NII_FILE" == *.nii.gz ]]; then
            if ! cp "$NII_FILE" "${RESOURCE_DIR}/${SCAN_TYPE}.nii.gz"; then
              warning_msg "Failed to copy file: $NII_FILE to ${RESOURCE_DIR}/${SCAN_TYPE}.nii.gz"
              continue
            fi
         else
            if ! cp "$NII_FILE" "${RESOURCE_DIR}/${SCAN_TYPE}.nii"; then
              warning_msg "Failed to copy file: $NII_FILE to ${RESOURCE_DIR}/${SCAN_TYPE}.nii"
              continue
            fi
         fi
      else
         echo "    No NIfTI file found in $DATE_DIR. Running dcm2niix conversion..."
         if ! dcm2niix -z y -f "${SCAN_TYPE}" -o "${RESOURCE_DIR}" "$DATE_DIR"; then
            warning_msg "dcm2niix conversion failed for directory: $DATE_DIR"
            continue
         fi
      fi
      echo ""
    done
  done
  echo ""
done

echo "Script completed. Some operations may have failed - check warnings above."
