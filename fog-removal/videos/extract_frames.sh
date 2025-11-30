#!/bin/bash

# --- CONFIGURATION ---
# The directory where the images are saved
OUTPUT_DIR="../images/processed_images"
# The temporary text file for FFmpeg
TEMP_LIST="inputs.txt"

# 1. Create the output directory if it doesn't exist
# mkdir -p creates parent directories if needed and doesn't error if it exists
echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# 2. Generate the input list dynamically
# We use a loop to safely handle filenames that might contain spaces
echo "Generating video list..."
if ls *.mp4 1> /dev/null 2>&1; then
    # Clear the file if it exists
    > "$TEMP_LIST"
    for f in *.mp4; do
        echo "file '$f'" >> "$TEMP_LIST"
    done
else
    echo "Error: No .mp4 files found in this directory."
    exit 1
fi

# 3. Run the FFmpeg command
echo "Starting extraction (1 frame every 5 seconds, Resizing to 1024x512)..."
ffmpeg -f concat -safe 0 -i "$TEMP_LIST" \
       -vf "fps=1/5,scale=1024:512" \
       -start_number 0 \
       "$OUTPUT_DIR/foggy_image_%d.png"

# 4. Cleanup the temporary text file
rm "$TEMP_LIST"

echo "Success! Processed images are in $OUTPUT_DIR"