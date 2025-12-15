#!/usr/bin/env python3
"""
Batch evaluation script for Faster R-CNN on multiple image folders.
Detector-safe rain preprocessing integrated.
"""

import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np
from tqdm import tqdm
import cv2
import sys
from io import StringIO
import os
from pathlib import Path
import shutil
from PIL import Image
import argparse


# =============================================================================
# DETECTOR-SAFE RAIN PREPROCESSING
# =============================================================================

def rain_preprocess_detector_safe(img, gamma=1.05):
    """
    Minimal, detector-safe rain preprocessing.
    Preserves texture and feature statistics.
    """

    if isinstance(img, Image.Image):
        img = np.array(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    img_f = img.astype(np.float32) / 255.0

    if gamma != 1.0:
        img_f = np.power(img_f, 1.0 / gamma)

    return np.clip(img_f * 255.0, 0, 255).astype(np.uint8)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base BDD100K has 10 classes, Rain fine-tuned has 9 classes (no 'train')
CLASSES_BDD = [
    'pedestrian',      # ID: 1
    'rider',           # ID: 2
    'car',             # ID: 3
    'truck',           # ID: 4
    'bus',             # ID: 5
    'train',           # ID: 6
    'motorcycle',      # ID: 7
    'bicycle',         # ID: 8
    'traffic light',   # ID: 9
    'traffic sign'     # ID: 10
]

CLASSES_RAIN = [
    'pedestrian',      # ID: 1
    'car',             # ID: 2
    'truck',           # ID: 3
    'bus',             # ID: 4
    'bicycle',         # ID: 5
    'motorcycle',      # ID: 6
    'rider',           # ID: 7
    'traffic light',   # ID: 8
    'traffic sign'     # ID: 9
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Faster R-CNN on different rain-processed datasets"
    )
    parser.add_argument(
        "processed_type",
        choices=["mca", "wavelet"],
        help="Which processed dataset to evaluate"
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "finetuned"],
        default="baseline",
        help="Which model to use: baseline (BDD100K) or finetuned (rain-optimized)"
    )
    return parser.parse_args()

# =============================================================================
# CLASS MAPPING: Map model predictions to ground truth category IDs
# =============================================================================
#
# Ground Truth (rain_annotations.json) uses this ordering:
#   1: pedestrian, 2: car, 3: truck, 4: bus, 5: bicycle,
#   6: motorcycle, 7: rider, 8: traffic light, 9: traffic sign
#
# The base BDD model was trained with a DIFFERENT class ordering, so we need
# to map its predictions to the correct ground truth category IDs.
# =============================================================================

# Mapping from base BDD model class indices to rain ground truth category IDs
BDD_TO_RAIN_MAPPING = {
    0: 1,  # pedestrian → pedestrian
    1: 7,  # rider → rider
    2: 2,  # car → car
    3: 3,  # truck → truck
    4: 4,  # bus → bus
    5: None,  # train → not in rain dataset (skip)
    6: 6,  # motorcycle → motorcycle
    7: 5,  # bicycle → bicycle
    8: 8,  # traffic light → traffic light
    9: 9,  # traffic sign → traffic sign
}

# Mapping from rain fine-tuned model class indices to ground truth category IDs
# (This is simple: just add 1, since they have the same ordering)
RAIN_TO_RAIN_MAPPING = {i: i + 1 for i in range(9)}

# =============================================================================
# MODEL SELECTION: Toggle between base model and fine-tuned model
# =============================================================================
#
# To switch between models, use the --model argument:
#   - baseline:  Use Base BDD100K Model (10 classes, includes 'train' class)
#   - finetuned: Use Rain Fine-tuned Model (9 classes, optimized for rain)
#
# The script will automatically use the correct class list, checkpoint, and mapping.
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = "./evaluation_results"

args = parse_args()

# Configure model based on command line argument
if args.model == "finetuned":
    CHECKPOINT_PATH = "../FasterRCNN/finetune_rain.pth"
    MODEL_NAME = "Rain Fine-tuned Model"
    SCORE_THRESHOLD = 0.50  # Optimized for fine-tuned model
    CLASSES = CLASSES_RAIN  # 9 classes
    CLASS_MAPPING = RAIN_TO_RAIN_MAPPING
else:  # baseline
    CHECKPOINT_PATH = "../FasterRCNN/finetune_bdd.pth"
    MODEL_NAME = "Base BDD100K Model"
    SCORE_THRESHOLD = 0.50  # Default threshold
    CLASSES = CLASSES_BDD  # 10 classes
    CLASS_MAPPING = BDD_TO_RAIN_MAPPING

if args.processed_type == "mca":
    processed_input_dir = "inputs_images_processed_mca"
elif args.processed_type == "wavelet":
    processed_input_dir = "inputs_images_processed_wavelet"
else:
    raise ValueError("Invalid processed_type")

FOLDER_CONFIGS = {
    "images_pre_processed": {
        "input_images": "./images_pre_processed",
        "annotations": "../temp_train_val_split/val_split.json",
        "output_images": "./output_unprocessed_images",
        "description": "Rain Removal - Unprocessed Images"
    },
    "images_processed": {
        "input_images": f"./{processed_input_dir}",
        "annotations": "../temp_train_val_split/val_split.json",
        "output_images": f"./output_processed_{args.processed_type}",
        "description": f"Rain Removal - Processed Images ({args.processed_type})"
    },
}

# =============================================================================
# UTILITIES
# =============================================================================

def cleanup_output_dirs(folder_configs, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for config in folder_configs.values():
        if os.path.exists(config['output_images']):
            shutil.rmtree(config['output_images'])

def setup_model(checkpoint_path, device='cuda'):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD
    cfg.MODEL.WEIGHTS = checkpoint_path

    # Verify weights are being loaded
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"\n⚠️  WARNING: Checkpoint path '{checkpoint_path}' does not exist!")
        print("Model will have RANDOM weights and produce no valid predictions.\n")
    else:
        print(f"✓ Loading weights from: {checkpoint_path}")
        print(f"  File size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB")
        print(f"  Number of classes: {len(CLASSES)}\n")

    return DefaultPredictor(cfg)

def setup_metadata():
    MetadataCatalog.get("bdd_eval").thing_classes = CLASSES
    return MetadataCatalog.get("bdd_eval")

# =============================================================================
# INFERENCE
# =============================================================================

def run_inference_and_save(
    predictor,
    annotations_path,
    input_image_dir,
    output_json,
    output_image_dir,
    metadata,
    save_viz=True
):
    if save_viz:
        os.makedirs(output_image_dir, exist_ok=True)

    sys.stdout = StringIO()
    coco_gt = COCO(annotations_path)
    sys.stdout = sys.__stdout__

    predictions = []
    image_ids = coco_gt.getImgIds()

    for img_id in tqdm(image_ids, desc="Running inference"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(input_image_dir, img_info['file_name'])

        image = cv2.imread(img_path)
        if image is None:
            continue

        # =====================================================
        # DROP-IN RAIN PREPROCESSING (THIS IS THE ONLY CHANGE)
        # =====================================================
        image = rain_preprocess_detector_safe(image, gamma=1.05)

        outputs = predictor(image)
        instances = outputs["instances"]

        boxes = instances.pred_boxes.tensor.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        classes = instances.pred_classes.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            cls_idx = int(cls)
            # Map model's class index to ground truth category ID
            category_id = CLASS_MAPPING.get(cls_idx)

            # Skip predictions for classes not in ground truth (e.g., 'train')
            if category_id is None:
                continue

            x1, y1, x2, y2 = box
            predictions.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score)
            })

        if save_viz:
            v = Visualizer(
                image[:, :, ::-1],
                metadata=metadata,
                scale=1.0
            )
            out = v.draw_instance_predictions(instances.to("cpu"))
            out_path = os.path.join(output_image_dir, img_info['file_name'])
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, out.get_image()[:, :, ::-1])

    with open(output_json, "w") as f:
        json.dump(predictions, f)

    return len(predictions), len(image_ids)

# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(annotations_path, predictions_json):
    # Check if predictions file is empty
    with open(predictions_json, 'r') as f:
        predictions = json.load(f)

    if len(predictions) == 0:
        print("\nWARNING: No predictions made! Model may not have valid weights loaded.")
        print("This typically happens when:")
        print("  - Model weights are not loaded (pretrained=False)")
        print("  - Score threshold is too high for untrained model")
        print("  - Checkpoint file is corrupted or invalid")
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AR": 0.0,
        }

    sys.stdout = StringIO()
    coco_gt = COCO(annotations_path)
    coco_dt = coco_gt.loadRes(predictions_json)
    sys.stdout = sys.__stdout__

    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "AP": coco_eval.stats[0] * 100,
        "AP50": coco_eval.stats[1] * 100,
        "AP75": coco_eval.stats[2] * 100,
        "AR": coco_eval.stats[6] * 100,
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    cleanup_output_dirs(FOLDER_CONFIGS, OUTPUT_DIR)

    metadata = setup_metadata()
    predictor = setup_model(CHECKPOINT_PATH, DEVICE)

    print("\n" + "=" * 80)
    print("STARTING EVALUATION")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Score Threshold: {SCORE_THRESHOLD}")
    print("=" * 80 + "\n")

    for folder_id, config in FOLDER_CONFIGS.items():
        print(f"\nEvaluating: {config['description']}")

        predictions_json = os.path.join(
            OUTPUT_DIR, f"{folder_id}_predictions.json"
        )

        num_preds, num_imgs = run_inference_and_save(
            predictor,
            config['annotations'],
            config['input_images'],
            predictions_json,
            config['output_images'],
            metadata
        )

        metrics = compute_metrics(
            config['annotations'],
            predictions_json
        )

        print(f"Images: {num_imgs}")
        print(f"Predictions: {num_preds}")
        print(f"AP: {metrics['AP']:.2f}")
        print(f"AP50: {metrics['AP50']:.2f}")
        print(f"AP75: {metrics['AP75']:.2f}")
        print(f"AR: {metrics['AR']:.2f}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print(f"Model used: {MODEL_NAME}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
