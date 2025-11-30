#!/bin/bash
# Setup script for Faster R-CNN with Detectron2

set -e

echo "Setting up Faster R-CNN with Detectron2..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found. Please install pip first."
    exit 1
fi

echo ""
echo "Step 1: Installing PyTorch..."
echo "Note: You may need to install PyTorch manually based on your system."
echo "Visit https://pytorch.org/get-started/locally/ for instructions."
echo ""
read -p "Press Enter to continue after installing PyTorch, or Ctrl+C to exit..."

echo ""
echo "Step 2: Installing Detectron2..."
echo ""
echo "Detectron2 must be built from source for Python 3.13+"
echo "This may take 10-20 minutes and requires:"
echo "  - C++ compiler (Xcode Command Line Tools on macOS)"
echo "  - CMake (install via: brew install cmake)"
echo ""
read -p "Press Enter to continue with installation, or Ctrl+C to install dependencies manually..."

# Check for required tools
if ! command -v cmake &> /dev/null; then
    echo "Warning: CMake not found. Install with: brew install cmake"
    echo "Continuing anyway..."
fi

# Try pre-built wheels first (for older Python versions)
echo "Attempting to install from pre-built wheels (may fail on Python 3.13+)..."
if pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/index.html 2>/dev/null; then
    echo "✓ Detectron2 installed successfully from pre-built wheels"
elif pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html 2>/dev/null; then
    echo "✓ Detectron2 installed successfully from CPU wheels"
else
    echo "Pre-built wheels not available. Building from source..."
    echo "This will take several minutes..."
    if pip3 install 'git+https://github.com/facebookresearch/detectron2.git'; then
        echo "✓ Detectron2 installed successfully from source"
    else
        echo ""
        echo "Error: Failed to install Detectron2"
        echo ""
        echo "Please install manually using one of these methods:"
        echo "  1. Build from source: pip3 install 'git+https://github.com/facebookresearch/detectron2.git'"
        echo "  2. For older Python: pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/index.html"
        echo ""
        echo "Make sure you have:"
        echo "  - C++ compiler (xcode-select --install on macOS)"
        echo "  - CMake (brew install cmake on macOS)"
        exit 1
    fi
fi

echo ""
echo "Step 3: Installing other dependencies..."
pip3 install -r requirements.txt

echo ""
echo "✓ Setup complete!"
echo ""
echo "You can now run inference with:"
echo "  python3 run_inference.py --image-dir /path/to/images --output-dir outputs"
echo ""

