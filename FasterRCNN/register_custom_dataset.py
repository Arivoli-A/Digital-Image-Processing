"""
Custom dataset registration for Detectron2.
Supports both COCO-format datasets (with annotations) and image-only datasets (for inference).
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def register_custom_coco_dataset(
    name: str,
    json_file: str,
    image_root: str,
    thing_classes: Optional[List[str]] = None,
):
    """
    Register a COCO-format dataset for training/evaluation.
    
    Args:
        name: Unique name for the dataset
        json_file: Path to COCO format JSON annotation file
        image_root: Path to directory containing images
        thing_classes: List of class names (if None, will try to infer from JSON)
    """
    from detectron2.data.datasets import register_coco_instances
    
    register_coco_instances(name, {}, json_file, image_root)
    
    # Set metadata if classes provided
    if thing_classes:
        MetadataCatalog.get(name).thing_classes = thing_classes


def register_image_folder_dataset(
    name: str,
    image_dir: str,
    thing_classes: Optional[List[str]] = None,
):
    """
    Register a folder of images for inference-only (no annotations).
    
    Args:
        name: Unique name for the dataset
        image_dir: Path to directory containing images
        thing_classes: List of class names (optional, for visualization)
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    
    # Supported image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    
    # Find all images
    image_files = sorted([
        str(p) for p in image_dir.rglob("*")
        if p.suffix.lower() in valid_exts and p.is_file()
    ])
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    def get_image_dicts():
        """Return list of dicts in Detectron2 format."""
        dataset_dicts = []
        for image_path in image_files:
            # Read image to get dimensions
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            height, width = img.shape[:2]
            
            record = {
                "file_name": image_path,
                "image_id": Path(image_path).stem,
                "height": height,
                "width": width,
            }
            dataset_dicts.append(record)
        return dataset_dicts
    
    DatasetCatalog.register(name, get_image_dicts)
    
    # Set metadata
    if thing_classes:
        MetadataCatalog.get(name).thing_classes = thing_classes
    else:
        # Default COCO classes if none provided
        MetadataCatalog.get(name).thing_classes = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]


def get_dataset_info(name: str) -> Dict[str, Any]:
    """Get information about a registered dataset."""
    if name not in DatasetCatalog.list():
        raise ValueError(f"Dataset '{name}' not registered")
    
    dataset_dicts = DatasetCatalog.get(name)
    metadata = MetadataCatalog.get(name)
    
    return {
        "num_images": len(dataset_dicts),
        "thing_classes": getattr(metadata, "thing_classes", None),
        "num_classes": len(getattr(metadata, "thing_classes", [])) if hasattr(metadata, "thing_classes") else 0,
    }


