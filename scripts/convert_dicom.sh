#!/bin/bash

# Check if input and output file name are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: convert_dicom.sh <input_dir> <output_dir> <output_file_name>"
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
OUTPUT_FILE=$3

# Run dcm2niix with user-defined output file name
dcm2niix -o "$OUTPUT_DIR" -f "$OUTPUT_FILE" "$INPUT_DIR"
