#!/bin/bash

# Folder ID from Google Drive
FOLDER_ID="1IMACwdW7y_gb4Iq5yWBQl93y1-FN3eeD"

OUTPUT_DIR="./raw_dataset"

# Install gdown if not installed
if ! command -v gdown &> /dev/null
then
    echo "gdown not found. Installing..."
    pip install gdown || exit 1
fi

# Download the folder
echo "Downloading folder..."
gdown "https://drive.google.com/drive/folders/${FOLDER_ID}" --folder -O "${OUTPUT_DIR}"

echo "Done!"
