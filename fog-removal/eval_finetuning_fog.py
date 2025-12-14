#!/usr/bin/env python3
"""
Fog Evaluation Pipeline: Training & Comparison
1. Splits Fog dataset into Train/Test.
2. Fine-tunes a model on the Train split.
3. Evaluates the fine-tuned model on Test data.
4. Compares against image processing methods (Kim, DCP).
"""

import torch
import json
import os
import sys
import cv2
import numpy as np
from datetime import datetime
from io import StringIO
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Detectron2 Imports
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

# COCO Tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =====================================================================
# CONFIGURATION
# =====================================================================

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

# General Settings
GLOBAL_CONFIG = {
    "base_model": "../FasterRCNN/untuned_model.pth",
    "output_dir": "../evaluation_results/fog",
    "viz_dir": "./fog_dataset/output_images/finetuned_images",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Training Parameters
    "train_split": 0.8,
    "finetune_epochs": 3,
    "finetune_batch_size": 4,
    "finetune_lr": 0.0001,
    "checkpoint_period": 50,
}

# Specific Fog Dataset Configuration
FOG_DATASET = {
    "raw_images": "./fog_dataset/input_images/unprocessed_images",
    "annotations": "./fog_dataset/labels.json",
    # Methods to compare against fine-tuning
    "processing_methods": {
        "kim_pipeline": "./fog_dataset/input_images/kim_pipeline_images",
        "dcp": "./fog_dataset/input_images/dcp_images",
    },
}

# =====================================================================
# UTILITY FUNCTIONS: DATASET HANDLING
# =====================================================================


def split_dataset(annotations_path, train_ratio=0.8):
    """Split dataset into train and test sets."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    coco = COCO(annotations_path)
    sys.stdout = old_stdout

    base_info = coco.dataset.get(
        "info",
        {
            "description": "Split Dataset",
            "date_created": datetime.now().isoformat(),
        },
    )
    base_licenses = coco.dataset.get("licenses", [])

    image_ids = coco.getImgIds()
    train_ids, test_ids = train_test_split(
        image_ids, train_size=train_ratio, random_state=42
    )

    def build_split(ids, split_name):
        imgs = coco.loadImgs(ids)
        anns = []
        for img_id in ids:
            anns.extend(coco.loadAnns(coco.getAnnIds(imgIds=img_id)))

        split_info = base_info.copy()
        split_info["description"] = f"{base_info.get('description')} ({split_name})"

        return {
            "info": split_info,
            "licenses": base_licenses,
            "images": imgs,
            "annotations": anns,
            "categories": coco.loadCats(coco.getCatIds()),
        }

    return (
        build_split(train_ids, "Train"),
        build_split(test_ids, "Test"),
        train_ids,
        test_ids,
    )


# =====================================================================
# UTILITY FUNCTIONS: INFERENCE & METRICS
# =====================================================================


def setup_predictor(weights_path, device="cuda", threshold=0.3):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = weights_path
    return DefaultPredictor(cfg)


def run_inference(
    predictor, annotations_path, image_dir, output_viz_dir=None, metadata=None
):
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    coco_gt = COCO(annotations_path)
    sys.stdout = old_stdout

    # Map Categories
    cat_name_to_id = {
        cat["name"]: cat["id"] for cat in coco_gt.loadCats(coco_gt.getCatIds())
    }
    class_idx_to_coco_id = {
        i: cat_name_to_id[c] for i, c in enumerate(CLASSES) if c in cat_name_to_id
    }

    predictions = []
    image_ids = coco_gt.getImgIds()

    desc = "      Inference"
    if output_viz_dir:
        desc += " (+Saving Visuals)"
        os.makedirs(output_viz_dir, exist_ok=True)

    for img_id in tqdm(image_ids, desc=desc, leave=False):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info["file_name"])

        try:
            image = cv2.imread(img_path)
            if image is None:
                continue

            outputs = predictor(image)
            instances = outputs["instances"]

            # Format predictions
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            classes_ids = instances.pred_classes.cpu().numpy()

            for box, score, cls in zip(boxes, scores, classes_ids):
                if int(cls) not in class_idx_to_coco_id:
                    continue
                predictions.append(
                    {
                        "image_id": img_id,
                        "category_id": class_idx_to_coco_id[int(cls)],
                        "bbox": [
                            float(box[0]),
                            float(box[1]),
                            float(box[2] - box[0]),
                            float(box[3] - box[1]),
                        ],
                        "score": float(score),
                    }
                )

            if output_viz_dir and metadata:
                v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imwrite(
                    os.path.join(output_viz_dir, img_info["file_name"]),
                    out.get_image()[:, :, ::-1],
                )

        except Exception:
            continue

    return predictions


def compute_metrics(annotations_path, predictions):
    keys = ["AP", "AP50", "AP75", "AR100"]
    if not predictions:
        return {k: 0.0 for k in keys}

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        coco_gt = COCO(annotations_path)
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        results = {
            "AP": float(stats[0] * 100),
            "AP50": float(stats[1] * 100),
            "AP75": float(stats[2] * 100),
            "AR100": float(stats[8] * 100),
        }
    except Exception:
        results = {k: 0.0 for k in keys}

    sys.stdout = old_stdout
    return results


# =====================================================================
# MAIN PIPELINE
# =====================================================================


def run_fog_pipeline():
    print(f"\n{'='*60}")
    print(f"STARTING FOG EVALUATION PIPELINE")
    print(f"{'='*60}")

    # 1. Setup Directories
    output_dir = GLOBAL_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    train_ann_path = os.path.join(output_dir, "train_annotations.json")
    test_ann_path = os.path.join(output_dir, "test_annotations.json")
    model_output_dir = os.path.join(output_dir, "finetuned_model")
    os.makedirs(model_output_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # PHASE 1: PREPARE DATA
    # -----------------------------------------------------------------
    print(f"  [Phase 1] Splitting Dataset ({GLOBAL_CONFIG['train_split']} split)...")
    train_data, test_data, train_ids, test_ids = split_dataset(
        FOG_DATASET["annotations"], train_ratio=GLOBAL_CONFIG["train_split"]
    )

    with open(train_ann_path, "w") as f:
        json.dump(train_data, f)
    with open(test_ann_path, "w") as f:
        json.dump(test_data, f)
    print(f"    ✓ Train: {len(train_ids)} | Test: {len(test_ids)}")

    # Register Train Set
    train_name = "fog_train"
    if train_name in DatasetCatalog.list():
        DatasetCatalog.remove(train_name)
        MetadataCatalog.remove(train_name)
    register_coco_instances(train_name, {}, train_ann_path, FOG_DATASET["raw_images"])

    # -----------------------------------------------------------------
    # PHASE 2: FINE-TUNE MODEL
    # -----------------------------------------------------------------
    print(f"  [Phase 2] Training Model...")

    iters = (len(train_ids) // GLOBAL_CONFIG["finetune_batch_size"]) * GLOBAL_CONFIG[
        "finetune_epochs"
    ]

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = GLOBAL_CONFIG["base_model"]
    cfg.SOLVER.IMS_PER_BATCH = GLOBAL_CONFIG["finetune_batch_size"]
    cfg.SOLVER.BASE_LR = GLOBAL_CONFIG["finetune_lr"]
    cfg.SOLVER.MAX_ITER = max(10, iters)  # Ensure at least some iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = GLOBAL_CONFIG["checkpoint_period"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
    cfg.OUTPUT_DIR = model_output_dir

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    print(f"    ✓ Model saved to: {final_model_path}")

    # -----------------------------------------------------------------
    # PHASE 3: EVALUATION & COMPARISON
    # -----------------------------------------------------------------
    print(f"  [Phase 3] Evaluation...")

    # Setup Metadata
    metadata_name = "fog_eval"
    if metadata_name not in MetadataCatalog.list():
        MetadataCatalog.get(metadata_name).thing_classes = CLASSES
    metadata = MetadataCatalog.get(metadata_name)

    results = {}

    # A. Evaluate Fine-Tuned
    print("    -> Testing Fine-Tuned Model...")
    viz_dir = GLOBAL_CONFIG["viz_dir"]
    predictor_ft = setup_predictor(final_model_path, GLOBAL_CONFIG["device"])
    preds_ft = run_inference(
        predictor_ft,
        test_ann_path,
        FOG_DATASET["raw_images"],
        output_viz_dir=viz_dir,
        metadata=metadata,
    )
    results["finetuning"] = compute_metrics(test_ann_path, preds_ft)

    # B. Evaluate Processing Methods
    print("    -> Testing Processing Baselines...")
    predictor_orig = setup_predictor(
        GLOBAL_CONFIG["base_model"], GLOBAL_CONFIG["device"]
    )

    for method_name, img_path in FOG_DATASET["processing_methods"].items():
        print(f"       Processing: {method_name}...")
        preds = run_inference(predictor_orig, test_ann_path, img_path)
        results[method_name] = compute_metrics(test_ann_path, preds)

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL FOG RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'AP':<8} {'AP50':<8} {'AP75':<8} {'AR100':<8}")
    print("-" * 55)

    best_ap = 0
    winner = ""

    for method, mets in results.items():
        if mets["AP"] > best_ap:
            best_ap = mets["AP"]
            winner = method
        print(
            f"{method:<20} {mets['AP']:<8.2f} {mets['AP50']:<8.2f} {mets['AP75']:<8.2f} {mets['AR100']:<8.2f}"
        )

    print("-" * 55)
    print(f"WINNER: {winner.upper()} (AP: {best_ap:.2f})")

    # Save JSON
    save_path = os.path.join(output_dir, "fog_summary.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    # Ensure Base Output exists
    os.makedirs(GLOBAL_CONFIG["output_dir"], exist_ok=True)
    run_fog_pipeline()
