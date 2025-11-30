#!/usr/bin/env python3
"""
Batch inference script for Faster R-CNN object detection using Detectron2.
Supports custom datasets with or without annotations.
"""
from __future__ import annotations

import argparse
import sys
import os
import ssl
from pathlib import Path
from typing import List, Optional

# Fix SSL certificate issues on macOS
try:
    import certifi
    import urllib.request
    # Set SSL certificate file
    os.environ['SSL_CERT_FILE'] = certifi.where()
    # Create SSL context with certifi certificates
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    # Monkey patch urllib to use our SSL context
    original_urlopen = urllib.request.urlopen
    def urlopen_with_cert(*args, **kwargs):
        if 'context' not in kwargs:
            kwargs['context'] = ssl_context
        return original_urlopen(*args, **kwargs)
    urllib.request.urlopen = urlopen_with_cert
except ImportError:
    # Fallback: disable SSL verification (less secure, but works)
    ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from PIL import Image
from tqdm import tqdm

# Import custom dataset registration
from register_custom_dataset import register_image_folder_dataset, get_dataset_info

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Faster R-CNN object detection on a folder of images using Detectron2."
    )
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing images for inference.",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to config file. If not provided, uses default Faster R-CNN config.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to model weights. If not provided, uses default pretrained weights.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save output images with detections.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Score threshold for detections (0.0 to 1.0).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: 'cpu', 'cuda', 'cuda:0', 'mps', or 'auto'.",
    )
    parser.add_argument(
        "--model",
        default="faster_rcnn_R_50_FPN_3x",
        help="Model architecture. Options: faster_rcnn_R_50_FPN_3x, faster_rcnn_R_101_FPN_3x, etc.",
    )
    parser.add_argument(
        "--dataset",
        default="coco",
        help="Dataset name for metadata (class names). Use 'coco' for COCO classes.",
    )
    parser.add_argument(
        "--save-format",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format.",
    )
    parser.add_argument(
        "--instance-mode",
        default="color",
        choices=["color", "segmentation", "binary_mask"],
        help="Visualization mode: 'color' (colored boxes), 'segmentation' (colored masks), 'binary_mask' (black/white masks).",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't draw labels on detections.",
    )
    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Don't draw confidence scores on detections.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> str:
    """Resolve device string to appropriate device."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_name


def list_images(folder: Path) -> List[Path]:
    """List all image files in folder recursively."""
    files = sorted([
        p for p in folder.rglob("*")
        if p.suffix.lower() in VALID_EXTS and p.is_file()
    ])
    return files


def setup_config(
    config_file: Optional[str],
    weights: Optional[str],
    model: str,
    score_thresh: float,
    device: str,
) -> tuple:
    """Set up Detectron2 configuration and predictor."""
    cfg = get_cfg()
    
    # Load config file if provided, otherwise use model zoo config
    if config_file:
        cfg.merge_from_file(config_file)
    else:
        # Use default Faster R-CNN config from model zoo
        try:
            from detectron2 import model_zoo
            config_path = f"COCO-Detection/{model}.yaml"
            cfg.merge_from_file(model_zoo.get_config_file(config_path))
        except Exception as e:
            raise RuntimeError(
                f"Could not load default config for {model}. "
                f"Please provide --config-file. Error: {e}"
            )
    
    # Set model weights
    if weights:
        cfg.MODEL.WEIGHTS = weights
    else:
        # Use default pretrained weights from model zoo
        try:
            from detectron2 import model_zoo
            weights_url = model_zoo.get_checkpoint_url(f"COCO-Detection/{model}.yaml")
            
            # Check if model is already cached locally
            from pathlib import Path
            cache_dir = Path.home() / ".cache" / "detectron2"
            if weights_url:
                model_filename = weights_url.split('/')[-1]
                cached_model = cache_dir / model_filename
                if cached_model.exists():
                    print(f"Using cached model: {cached_model}")
                    cfg.MODEL.WEIGHTS = str(cached_model)
                else:
                    print(f"Downloading model weights (this may take a few minutes)...")
                    print(f"URL: {weights_url}")
                    print(f"Model will be cached at: {cache_dir}")
                    cfg.MODEL.WEIGHTS = weights_url
            else:
                raise RuntimeError("Could not get model weights URL")
        except Exception as e:
            raise RuntimeError(
                f"Could not load default weights for {model}. "
                f"Please provide --weights or run: python download_model.py {model}. "
                f"Error: {e}"
            )
    
    # Set score threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    
    # Set device
    cfg.MODEL.DEVICE = device
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    return cfg, predictor


def get_visualizer_mode(instance_mode: str) -> ColorMode:
    """Convert instance mode string to ColorMode enum."""
    mode_map = {
        "color": ColorMode.IMAGE,
        "segmentation": ColorMode.SEGMENTATION,
        "binary_mask": ColorMode.IMAGE_BW,
    }
    return mode_map.get(instance_mode, ColorMode.IMAGE)


def run_inference() -> None:
    """Main inference function."""
    args = parse_args()
    
    # Validate inputs
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise SystemExit(f"Image directory not found: {image_dir}")
    
    image_files = list_images(image_dir)
    if not image_files:
        raise SystemExit(f"No images found in {image_dir}")
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Resolve device
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    
    # Register custom dataset if needed (for metadata)
    dataset_name = f"custom_inference_{args.dataset}"
    try:
        # Try to register the image folder
        register_image_folder_dataset(dataset_name, str(image_dir))
        dataset_info = get_dataset_info(dataset_name)
        print(f"Registered dataset '{dataset_name}' with {dataset_info['num_images']} images")
    except Exception as e:
        print(f"Warning: Could not register custom dataset: {e}")
        print("Using default COCO metadata")
        dataset_name = args.dataset
    
    # Setup config and predictor
    try:
        cfg, predictor = setup_config(
            args.config_file,
            args.weights,
            args.model,
            args.score_thresh,
            device,
        )
    except Exception as e:
        raise SystemExit(f"Failed to setup model: {e}")
    
    # Get metadata
    try:
        metadata = MetadataCatalog.get(dataset_name)
    except KeyError:
        # Fallback to COCO metadata
        metadata = MetadataCatalog.get("coco_2017_val")
        print("Using COCO metadata as fallback")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine visualization mode
    vis_mode = get_visualizer_mode(args.instance_mode)
    
    print(f"\nRunning inference on {len(image_files)} images...")
    print(f"Score threshold: {args.score_thresh}")
    print(f"Output directory: {output_dir}")
    
    # Run inference
    with torch.no_grad():
        for image_path in tqdm(image_files, desc="Processing images", unit="img"):
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path}, skipping")
                continue
            
            # Run prediction
            outputs = predictor(image)
            
            # Visualize
            visualizer = Visualizer(
                image[:, :, ::-1],  # Convert BGR to RGB
                metadata=metadata,
                scale=1.0,
                instance_mode=vis_mode,
            )
            
            # Get instances for visualization and summary
            instances = outputs["instances"].to("cpu")
            
            # Draw predictions
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
            
            # Get visualized image
            vis_image = vis_output.get_image()[:, :, ::-1]  # Convert RGB back to BGR
            
            # Save output
            rel_path = image_path.relative_to(image_dir)
            stem = rel_path.stem
            output_path = output_dir / f"{stem}_detections.{args.save_format}"
            
            # Create parent directories if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image
            cv2.imwrite(str(output_path), vis_image)
            
            # Print detection summary
            num_detections = len(instances)
            if num_detections > 0:
                classes = instances.pred_classes.cpu().numpy()
                scores = instances.scores.cpu().numpy()
                class_names = [metadata.thing_classes[c] for c in classes]
                print(f"\n{image_path.name}: {num_detections} detections")
                for cls_name, score in zip(class_names, scores):
                    print(f"  - {cls_name}: {score:.2f}")
    
    print(f"\nInference complete! Results saved to {output_dir}")


if __name__ == "__main__":
    run_inference()

