# Quick Start Guide

## Installation

1. **Install PyTorch** (if not already installed):
   ```bash
   # Visit https://pytorch.org/get-started/locally/ for your system
   pip install torch torchvision
   ```

2. **Install Detectron2** (IMPORTANT: Cannot use regular pip):
   
   **For Python 3.13+ (your case):**
   ```bash
   # Build from source (takes 10-20 minutes)
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```
   
   **Prerequisites for building from source:**
   ```bash
   # On macOS, install Xcode Command Line Tools:
   xcode-select --install
   
   # Install CMake:
   brew install cmake
   ```
   
   **Or use the installation script:**
   ```bash
   ./install_detectron2.sh
   ```

3. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

**Note**: Detectron2 is NOT in requirements.txt because it requires special installation. Install it separately first!

## Running Inference

### Basic Usage

Run object detection on a folder of images:

```bash
python run_inference.py --image-dir /path/to/images --output-dir outputs
```

### Example with Custom Images

```bash
# Using images from CCNet's custom dataset
python run_inference.py \
    --image-dir ../CCNet/datasets/custom/images \
    --output-dir ../CCNet/datasets/custom/outputs \
    --score-thresh 0.5
```

### Advanced Options

```bash
python run_inference.py \
    --image-dir /path/to/images \
    --output-dir outputs \
    --model faster_rcnn_R_101_FPN_3x \
    --score-thresh 0.6 \
    --device cuda \
    --instance-mode segmentation
```

## What You Get

- **Input**: Folder of images (JPG, PNG, etc.)
- **Output**: Images with bounding boxes, labels, and confidence scores
- **Format**: `{image_name}_detections.png`

## Training (Optional)

If you have COCO-format annotations:

```bash
python train_custom.py \
    --train-json annotations/train.json \
    --train-images images/train \
    --num-classes 3 \
    --thing-classes person car bicycle \
    --max-iter 1000
```

## Troubleshooting

- **No detections?** Lower `--score-thresh` (e.g., 0.3)
- **Out of memory?** Use smaller model: `--model faster_rcnn_R_50_FPN_3x`
- **Installation issues?** See main README.md for detailed instructions

