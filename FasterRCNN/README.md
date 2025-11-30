# Faster R-CNN with Detectron2

This directory contains a complete setup for running Faster R-CNN object detection using Facebook AI Research's [Detectron2](https://github.com/facebookresearch/detectron2) framework.

## Features

- **Custom Dataset Support**: Run inference on any folder of images without requiring annotations
- **Pre-trained Models**: Use state-of-the-art pre-trained Faster R-CNN models from the Detectron2 Model Zoo
- **Flexible Configuration**: Support for various model architectures and configurations
- **Easy Inference**: Simple command-line interface for batch processing images

## Installation

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended) or CPU
- pip

### Step 1: Install PyTorch

First, install PyTorch according to your system. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the appropriate installation command.

For example, on Linux with CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install Detectron2

**Important**: Detectron2 cannot be installed via regular pip. You must use one of the methods below.

**For Python 3.13 or newer versions (recommended method):**

Build from source (works on all platforms):
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**For older Python versions (3.7-3.12):**

**Option A: Pre-built wheel (Linux with CUDA)**
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/index.html
```

**Option B: Pre-built wheel (macOS CPU only)**
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/index.html
```

**Option C: Build from source (if pre-built wheels don't work)**
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**Note**: Building from source requires:
- C++ compiler (Xcode Command Line Tools on macOS: `xcode-select --install`)
- CMake (install via `brew install cmake` on macOS)
- May take 10-20 minutes to compile

### Step 3: Install Other Dependencies

```bash
cd FasterRCNN
pip install -r requirements.txt
```

## Quick Start

### Running Inference on Custom Images

The simplest way to use this setup is to run inference on a folder of images:

```bash
python run_inference.py \
    --image-dir /path/to/your/images \
    --output-dir outputs \
    --score-thresh 0.5
```

This will:
1. Load a pre-trained Faster R-CNN model (default: R50-FPN)
2. Process all images in the specified directory
3. Save detection results to the output directory

### Example with Custom Model

```bash
python run_inference.py \
    --image-dir datasets/custom/images \
    --output-dir datasets/custom/outputs \
    --model faster_rcnn_R_101_FPN_3x \
    --score-thresh 0.6 \
    --weights path/to/custom/weights.pth
```

## Usage

### Command-Line Arguments

```
--image-dir          (required) Directory containing images for inference
--output-dir          Output directory for results (default: outputs)
--config-file         Path to custom config file (optional, uses model zoo default)
--weights             Path to model weights (optional, downloads from model zoo)
--score-thresh        Detection confidence threshold 0.0-1.0 (default: 0.5)
--device              Device: cpu, cuda, cuda:0, mps, or auto (default: auto)
--model               Model architecture (default: faster_rcnn_R_50_FPN_3x)
--dataset             Dataset name for metadata (default: coco)
--save-format         Output format: png, jpg, jpeg (default: png)
--instance-mode       Visualization: color, segmentation, binary_mask (default: color)
--no-labels           Don't draw class labels on detections
--no-scores           Don't draw confidence scores on detections
```

### Available Models

Common Faster R-CNN models available in Detectron2 Model Zoo:

- `faster_rcnn_R_50_FPN_3x` - ResNet-50 backbone (default, good balance)
- `faster_rcnn_R_101_FPN_3x` - ResNet-101 backbone (more accurate, slower)
- `faster_rcnn_X_101_32x8d_FPN_3x` - ResNeXt-101 backbone (most accurate, slowest)

### Dataset Structure

For inference-only (no annotations needed):

```
your_dataset/
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

The script will automatically:
- Find all images recursively in the directory
- Support common formats: JPG, JPEG, PNG, BMP, TIF, TIFF
- Process them in batch

## Custom Dataset Registration

### For Inference-Only Images

The `register_custom_dataset.py` module automatically handles image-only datasets. The inference script registers your image folder automatically.

### For Training (COCO Format)

If you have annotations in COCO format, you can register them:

```python
from register_custom_dataset import register_custom_coco_dataset

register_custom_coco_dataset(
    name="my_dataset_train",
    json_file="annotations/train.json",
    image_root="images/train",
    thing_classes=["person", "car", "bicycle"]  # Your class names
)
```

## Output Format

The inference script saves images with:
- **Bounding boxes** around detected objects
- **Class labels** (e.g., "person", "car")
- **Confidence scores** (e.g., "0.95")
- **Color-coded** by class

Output files are named: `{original_name}_detections.{format}`

## Training on Custom Datasets

To train on your own dataset:

1. **Prepare annotations** in COCO format
2. **Register your dataset** using `register_custom_dataset.py`
3. **Configure training** using Detectron2's config system
4. **Train** using Detectron2's training tools

Example training script structure:

```python
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from register_custom_dataset import register_custom_coco_dataset

# Register dataset
register_custom_coco_dataset("my_train", "train.json", "train_images")

# Setup config
cfg = get_cfg()
cfg.merge_from_file("path/to/config.yaml")
cfg.DATASETS.TRAIN = ("my_train",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Your number of classes
cfg.MODEL.WEIGHTS = "path/to/pretrained.pth"

# Train
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

## Troubleshooting

### Installation Issues

**Problem**: `detectron2` installation fails
- **Solution**: Make sure you have the correct PyTorch version installed first
- Try building from source if pre-built wheels don't work

**Problem**: CUDA out of memory
- **Solution**: Use a smaller model (R_50 instead of R_101) or process fewer images at once

### Runtime Issues

**Problem**: No detections found
- **Solution**: Lower the `--score-thresh` value (e.g., 0.3)

**Problem**: Model downloads are slow
- **Solution**: Pre-download weights manually and use `--weights` to point to local file

## Resources

- [Detectron2 Documentation](https://detectron2.readthedocs.io/)
- [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)
- [Detectron2 GitHub](https://github.com/facebookresearch/detectron2)
- [COCO Dataset Format](https://cocodataset.org/#format-data)

## License

This code follows the same license as Detectron2 (Apache 2.0).

