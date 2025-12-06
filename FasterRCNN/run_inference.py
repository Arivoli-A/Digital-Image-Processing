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

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer

from register_custom_dataset import register_custom_coco_dataset

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

    parser.add_argument(
        "--annotation",
        default=None,
        help="Annotation json file for dataset",
    )
    
    parser.add_argument(
        "--thing-classes",
        nargs="+",
        help="List of class names (optional, for metadata)",
    )

    parser.add_argument(
    "--num-classes",
    type=int,
    default=80,
    help="Number of object classes (excluding background)",
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
            weights_url = model_zoo.get_checkpoint_url(f"COCO-Detection/{model}.yaml") #  'https://dl.cv.ethz.ch/bdd100k/det/models/faster_rcnn_r50_fpn_3x_det_bdd100k.pth' 
            
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

    # Set number of classes
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    
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
    dataset_name_coco = f"custom_inference_{args.dataset}_coco"
    
    try:
        # Try to register the image folder
        register_image_folder_dataset(dataset_name, str(image_dir)) #, thing_classes = args.thing_classes)
        dataset_info = get_dataset_info(dataset_name)
        print(f"Registered dataset '{dataset_name}' with {dataset_info['num_images']} images")

        # Only register COCO if annotation is provided
        if args.annotation:
            register_custom_coco_dataset(
                name=dataset_name_coco,
                json_file=args.annotation,
                image_root=image_dir,
                thing_classes=args.thing_classes
            )
            print(f"Registered annotation dataset '{dataset_name_coco}'")
        else:
            dataset_name_coco = None
            print("No annotation provided -> Evaluation metrics disabled.")
            
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

            if args.annotation:
                from pycocotools.coco import COCO
                coco = COCO(args.annotation)
                
                img_id = None
                if image_path.stem.isdigit():
                    img_ids_list = coco.getImgIds(imgIds=[int(image_path.stem)])
                    if len(img_ids_list) > 0:
                        img_id = img_ids_list[0]
                
                if img_id is not None:
                    gt_image = image.copy()
                    ann_ids = coco.getAnnIds(imgIds=[img_id])
                    anns = coco.loadAnns(ann_ids)
                    for ann in anns:
                        x, y, w, h = map(int, ann['bbox'])
                        cv2.rectangle(gt_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        class_name = coco.loadCats(ann['category_id'])[0]['name']
                        cv2.putText(gt_image, class_name, (x, y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Resize vis_image to have the same height as gt_image
                    if gt_image.shape[0] != vis_image.shape[0]:
                        new_width = int(vis_image.shape[1] * (gt_image.shape[0] / vis_image.shape[0]))
                        vis_image = cv2.resize(vis_image, (new_width, gt_image.shape[0]))
                    
                    combined = np.concatenate([gt_image, vis_image], axis=1)
                    
                    combined_output_path = output_dir / f"{image_path.stem}_gt_pred.{args.save_format}"
                    print("Saving combined GT & Pred image at", combined_output_path)
                    cv2.imwrite(str(combined_output_path), combined)

    print(f"\nInference complete! Results saved to {output_dir}")


    #  EVALUATION METRICS (ONLY IF ANNOTATION EXISTS)
   
    if dataset_name_coco is not None:
        print("\nRunning evaluation on dataset with annotation...")
        evaluator = COCOEvaluator(dataset_name_coco, cfg, False, output_dir=str(output_dir))
        val_loader = build_detection_test_loader(cfg, dataset_name_coco)
        
        #  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80 #len(args.thing_classes)

        # meta = MetadataCatalog.get(dataset_name_coco)
        
        # from pycocotools.coco import COCO
        
        # coco = COCO(args.annotation)
        # cats = coco.loadCats(coco.getCatIds())
        # cat_ids = [c['id'] for c in sorted(cats, key=lambda x: x['id'])]  # JSON IDs in order
        # names = [c['name'] for c in sorted(cats, key=lambda x: x['id'])]
        
        # meta.thing_classes = names
        # meta.thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(cat_ids)}
        
        # model = DefaultTrainer.build_model(cfg) 

        model = predictor.model
        model.eval()

        
        # for batch in val_loader:
        #     outputs = model(batch)
        #     print(batch)          # GT annotations
        #     print(outputs)        # Model predictions
    
      
        # print("Evaluator contiguous class order (0-based):")
        # for i, name in enumerate(meta.thing_classes):
        #     print(i, "→", name)

        import sys
        from contextlib import redirect_stdout
        
        
        output_file = Path(args.output_dir) / "detection_metrics_full.txt"
        
        with open(output_file, "w") as f:
            with redirect_stdout(f):
                results = inference_on_dataset(model, val_loader, evaluator)
        
                # results = inference_on_dataset(model, val_loader, evaluator)
            
                print("\n===== DETECTION METRICS =====")
                print(results)
        print(f"Full detection metrics saved to {output_file}")


    else:
        print("\nNo annotation provided: skipping evaluation metrics.")


if __name__ == "__main__":
    run_inference()

