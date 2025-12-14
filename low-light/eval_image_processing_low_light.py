#!/usr/bin/env python3
"""
Batch evaluation script for Faster R-CNN on multiple image folders.
Each folder has its own images and corresponding annotations.
Includes visualization output for each prediction.
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

# BDD100K classes
CLASSES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

# Configuration
CHECKPOINT_PATH = "../FasterRCNN/untuned_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "../evaluation_results/low-light"

# Folder configurations
# Format: folder_id: {"input_images": path, "annotations": path, "output_images": path, "description": text}
FOLDER_CONFIGS = {
    "low_light_unprocessed": {
        "input_images": "./ground_truth/test",
        "annotations": "./ground_truth/test/instances_default_test_bdd100k.json",
        "output_images": "./outputs/ground_truth/",
        "description": "Low Light - Unprocessed Images",
    },
    "low_light_bimef": {
        "input_images": "./BIMEF/",
        "annotations": "./BIMEF/instances_default_processed_bdd100k.json",
        "output_images": "./outputs/BIMEF",
        "description": "BIMEF",
    },
    "low_light_lime": {
        "input_images": "./LIME",
        "annotations": "./BIMEF/instances_default_processed_bdd100k.json",
        "output_images": "./outputs/LIME",
        "description": "LIME",
    },
}


def setup_model(checkpoint_path, device="cuda"):
    """Setup the detection model"""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.WEIGHTS = checkpoint_path

    predictor = DefaultPredictor(cfg)
    return predictor


def setup_metadata():
    """Setup metadata for visualization"""
    MetadataCatalog.get("bdd_eval").thing_classes = CLASSES
    return MetadataCatalog.get("bdd_eval")


def run_inference_and_save(
    predictor,
    annotations_path,
    input_image_dir,
    output_json,
    output_image_dir,
    metadata,
    save_viz=True,
):
    """Run inference on all images, save predictions and visualizations"""

    # Create output image directory
    if save_viz:
        os.makedirs(output_image_dir, exist_ok=True)

    # Suppress COCO loading output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    coco_gt = COCO(annotations_path)
    sys.stdout = old_stdout

    image_ids = coco_gt.getImgIds()
    predictions = []
    viz_count = 0

    for img_id in tqdm(image_ids, desc="  Running inference", leave=False):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(input_image_dir, img_info["file_name"])

        try:
            image = cv2.imread(img_path)
            if image is None:
                continue

            outputs = predictor(image)

            instances = outputs["instances"]
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()

            # Save predictions
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                predictions.append(
                    {
                        "image_id": img_id,
                        "category_id": int(cls) + 1,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )

            # Save visualization
            if save_viz:
                v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                # Save with original filename
                output_path = os.path.join(output_image_dir, img_info["file_name"])
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
                viz_count += 1

        except Exception as e:
            print(f"    Warning: Error processing image {img_id}: {e}")

    # Save predictions JSON
    with open(output_json, "w") as f:
        json.dump(predictions, f)

    return len(predictions), len(image_ids), viz_count


def compute_metrics(annotations_path, predictions_json):
    """Compute AP and AR metrics for all classes"""

    # Suppress COCO output
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    coco_gt = COCO(annotations_path)
    coco_dt = coco_gt.loadRes(predictions_json)

    sys.stdout = old_stdout

    results = {}
    all_metrics = {}

    for cat_id, cat_name in enumerate(CLASSES, start=1):
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            sys.stdout = old_stdout

            if coco_eval.stats is not None and len(coco_eval.stats) >= 12:
                all_metrics[cat_name] = {
                    # Precision metrics (AP)
                    "AP": float(coco_eval.stats[0] * 100),
                    "AP50": float(coco_eval.stats[1] * 100),
                    "AP75": float(coco_eval.stats[2] * 100),
                    "APs": float(coco_eval.stats[3] * 100),
                    "APm": float(coco_eval.stats[4] * 100),
                    "APl": float(coco_eval.stats[5] * 100),
                    # Recall metrics (AR)
                    "AR1": float(coco_eval.stats[6] * 100),  # maxDets=1
                    "AR10": float(coco_eval.stats[7] * 100),  # maxDets=10
                    "AR100": float(coco_eval.stats[8] * 100),  # maxDets=100
                    "ARs": float(coco_eval.stats[9] * 100),  # small objects
                    "ARm": float(coco_eval.stats[10] * 100),  # medium objects
                    "ARl": float(coco_eval.stats[11] * 100),  # large objects
                }
            else:
                all_metrics[cat_name] = {
                    k: 0.0
                    for k in [
                        "AP",
                        "AP50",
                        "AP75",
                        "APs",
                        "APm",
                        "APl",
                        "AR1",
                        "AR10",
                        "AR100",
                        "ARs",
                        "ARm",
                        "ARl",
                    ]
                }
        except Exception as e:
            sys.stdout = old_stdout
            all_metrics[cat_name] = {
                k: 0.0
                for k in [
                    "AP",
                    "AP50",
                    "AP75",
                    "APs",
                    "APm",
                    "APl",
                    "AR1",
                    "AR10",
                    "AR100",
                    "ARs",
                    "ARm",
                    "ARl",
                ]
            }

    # Organize results - now includes recall metrics
    metric_names = [
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
    ]

    for metric_name in metric_names:
        per_class = {}
        all_class_values = []

        for cat_name in CLASSES:
            value = max(0.0, all_metrics[cat_name][metric_name])
            per_class[cat_name] = value
            all_class_values.append(value)

        overall = float(np.mean(all_class_values))
        results[metric_name] = {"per_class": per_class, "overall": overall}

    return results


def evaluate_all_folders(predictor, metadata, save_visualizations=True):
    """Evaluate all folder configurations"""

    all_results = {}

    print("=" * 80)
    print("STARTING BATCH EVALUATION")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Folders to evaluate: {len(FOLDER_CONFIGS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Save visualizations: {save_visualizations}")
    print()

    for folder_id, config in FOLDER_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"Processing: {config['description']} [{folder_id}]")
        print(f"{'='*80}")
        print(f"  Input images: {config['input_images']}")
        print(f"  Annotations: {config['annotations']}")
        print(f"  Output images: {config['output_images']}")

        # Define output paths
        predictions_json = os.path.join(OUTPUT_DIR, f"{folder_id}_predictions.json")

        # Run inference and save visualizations
        num_preds, num_images, viz_count = run_inference_and_save(
            predictor,
            config["annotations"],
            config["input_images"],
            predictions_json,
            config["output_images"],
            metadata,
            save_viz=save_visualizations,
        )
        print(f"  ✓ Generated {num_preds} predictions from {num_images} images")
        if save_visualizations:
            print(f"  ✓ Saved {viz_count} visualization images")

        # Compute metrics
        print(f"  Computing metrics...")
        results = compute_metrics(config["annotations"], predictions_json)
        print(f"  ✓ Metrics computed")

        # Store results
        all_results[folder_id] = {
            "description": config["description"],
            "input_images_directory": config["input_images"],
            "annotations_file": config["annotations"],
            "output_images_directory": config["output_images"],
            "num_images": num_images,
            "num_predictions": num_preds,
            "num_visualizations": viz_count if save_visualizations else 0,
            "metrics": results,
        }

        print(f"  ✓ AP: {results['AP']['overall']:.2f}")
        print(f"  ✓ AP50: {results['AP50']['overall']:.2f}")
        print(f"  ✓ AP75: {results['AP75']['overall']:.2f}")

    return all_results


def save_results(all_results):
    """Save all results to JSON"""

    output_json = os.path.join(OUTPUT_DIR, "all_results.json")

    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ All results saved to {output_json}")

    # Create summary
    summary = []
    for folder_id, data in all_results.items():
        summary.append(
            {
                "folder_id": folder_id,
                "description": data["description"],
                "num_images": data["num_images"],
                "num_predictions": data["num_predictions"],
                "num_visualizations": data["num_visualizations"],
                "AP": data["metrics"]["AP"]["overall"],
                "AP50": data["metrics"]["AP50"]["overall"],
                "AP75": data["metrics"]["AP75"]["overall"],
                "AR1": data["metrics"]["AR1"]["overall"],
                "AR10": data["metrics"]["AR10"]["overall"],
                "AR100": data["metrics"]["AR100"]["overall"],
            }
        )

    summary_json = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved to {summary_json}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(
        f"{'Folder':<35} {'Imgs':<6} {'Preds':<7} {'AP':<7} {'AP50':<7} {'AP75':<7} {'AR1':<7} {'AR10':<7} {'AR100':<7}"
    )
    print("-" * 80)
    for item in summary:
        print(
            f"{item['description']:<35} {item['num_images']:<6} {item['num_predictions']:<7} "
            f"{item['AP']:<7.2f} {item['AP50']:<7.2f} {item['AP75']:<7.2f} "
            f"{item['AR1']:<7.2f} {item['AR10']:<7.2f} {item['AR100']:<7.2f}"
        )


def main():
    """Main evaluation pipeline"""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup metadata for visualization
    metadata = setup_metadata()

    # Load model once
    print("Loading model...")
    predictor = setup_model(CHECKPOINT_PATH, device=DEVICE)
    print("✓ Model loaded\n")

    # Evaluate all folders (set save_visualizations=False to skip viz)
    all_results = evaluate_all_folders(predictor, metadata, save_visualizations=True)

    # Save results
    save_results(all_results)

    print("\n" + "=" * 80)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
