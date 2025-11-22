#!/bin/bash

# Exit on error
set -e

# --- CONFIGURATION ---
# Path to compiled pylime.so
SRC_LIB="lib/pylime.cpython-311-x86_64-linux-gnu.so"

# Destination folder
DEST_DIR=".."

# Copy the file
cp "$SRC_LIB" "$DEST_DIR"


