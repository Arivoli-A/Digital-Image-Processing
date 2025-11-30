#!/bin/bash
# Standalone script to install Detectron2
# Use this if the main setup script fails

set -e

echo "Detectron2 Installation Script"
echo "================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check for required tools
echo ""
echo "Checking for required tools..."

if ! command -v cmake &> /dev/null; then
    echo "✗ CMake not found"
    echo "  Install with: brew install cmake"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ CMake found"
fi

# Check for C++ compiler
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "✗ C++ compiler not found"
    echo "  On macOS, install Xcode Command Line Tools: xcode-select --install"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ C++ compiler found"
fi

echo ""
echo "Installing Detectron2..."
echo "This may take 10-20 minutes..."
echo ""

# Try pre-built wheels first (for older Python versions)
echo "Attempting pre-built wheels..."
if pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/index.html 2>/dev/null; then
    echo ""
    echo "✓ Successfully installed Detectron2 from pre-built wheels!"
    exit 0
fi

# Try CPU wheels
if pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html 2>/dev/null; then
    echo ""
    echo "✓ Successfully installed Detectron2 from CPU wheels!"
    exit 0
fi

# Build from source
echo "Pre-built wheels not available. Building from source..."
echo "This will take 10-20 minutes..."
echo ""

if pip3 install 'git+https://github.com/facebookresearch/detectron2.git'; then
    echo ""
    echo "✓ Successfully installed Detectron2 from source!"
    exit 0
else
    echo ""
    echo "✗ Installation failed"
    echo ""
    echo "Troubleshooting:"
    echo "1. Make sure you have all dependencies:"
    echo "   - PyTorch: pip3 install torch torchvision"
    echo "   - CMake: brew install cmake"
    echo "   - C++ compiler: xcode-select --install"
    echo ""
    echo "2. Try installing in a virtual environment:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install torch torchvision"
    echo "   pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    echo ""
    exit 1
fi


