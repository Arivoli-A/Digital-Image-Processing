#!/usr/bin/env python3
"""
Fine-tune Faster R-CNN on Rain Dataset

This script fine-tunes the Faster R-CNN model on a custom dataset.
By default, it uses the rain-removal dataset.

Usage:
    python finetune_rcnn_rain.py
"""

import torch
import os
import json
import random
import shutil
from pathlib import Path
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger

setup_logger()

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_CONFIG = {
    "train_images": "./ground_truth/train",
    "train_annotations": "./ground_truth/train/instances_default_train_bdd100k.json",
    "val_images": "./ground_truth/test",
    "val_annotations": "./ground_truth/test/instances_default_test_bdd100k.json",
    "val_split_ratio": 0.2,
}

TRAINING_CONFIG = {
    "base_checkpoint": "../FasterRCNN/untuned_model.pth",
    "output_dir": "../FasterRCNN/low_light_finetuned",
    "batch_size": 2,
    "learning_rate": 0.00025,  # Increased for better convergence
    "max_iterations": 3000,  # Increased significantly
    "checkpoint_period": 3001,
    "eval_period": 300,
    "score_threshold": 0.3,  # More reasonable threshold
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

FINETUNED_MODEL_NAME = "finetune_lowlight.pth"

# =============================================================================
# FUNCTIONS
# =============================================================================


def create_train_val_split(
    annotations_path, images_dir, val_ratio=0.2, output_dir="./temp_split"
):
    """Create train/val split from a single annotation file."""
    with open(annotations_path, "r") as f:
        coco_data = json.load(f)

    image_ids = [img["id"] for img in coco_data["images"]]
    random.shuffle(image_ids)

    split_idx = int(len(image_ids) * (1 - val_ratio))
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    train_data = {
        "info": coco_data["info"],
        "licenses": coco_data["licenses"],
        "categories": coco_data["categories"],
        "images": [img for img in coco_data["images"] if img["id"] in train_ids],
        "annotations": [
            ann for ann in coco_data["annotations"] if ann["image_id"] in train_ids
        ],
    }

    val_data = {
        "info": coco_data["info"],
        "licenses": coco_data["licenses"],
        "categories": coco_data["categories"],
        "images": [img for img in coco_data["images"] if img["id"] in val_ids],
        "annotations": [
            ann for ann in coco_data["annotations"] if ann["image_id"] in val_ids
        ],
    }

    os.makedirs(output_dir, exist_ok=True)
    train_json = os.path.join(output_dir, "train_split.json")
    val_json = os.path.join(output_dir, "val_split.json")

    with open(train_json, "w") as f:
        json.dump(train_data, f)

    with open(val_json, "w") as f:
        json.dump(val_data, f)

    print(f"Split complete:")
    print(
        f"  Training: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations"
    )
    print(
        f"  Validation: {len(val_data['images'])} images, {len(val_data['annotations'])} annotations"
    )

    return train_json, val_json


def clear_datasets():
    """Clear any existing dataset registrations."""
    print("Clearing existing dataset registrations...")
    for dataset_name in ["rain_train", "rain_val"]:
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
        if dataset_name in MetadataCatalog.list():
            MetadataCatalog.remove(dataset_name)
    print("Cleanup complete!\n")


def setup_training_config(training_config, num_classes, device="cuda"):
    """Setup Detectron2 configuration for fine-tuning."""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    cfg.DATASETS.TRAIN = ("rain_train",)
    cfg.DATASETS.TEST = ("rain_val",)

    cfg.MODEL.WEIGHTS = training_config["base_checkpoint"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = training_config["score_threshold"]
    cfg.MODEL.DEVICE = device

    # Solver settings
    cfg.SOLVER.IMS_PER_BATCH = training_config["batch_size"]
    cfg.SOLVER.BASE_LR = training_config["learning_rate"]
    cfg.SOLVER.MAX_ITER = training_config["max_iterations"]
    cfg.SOLVER.STEPS = (
        int(training_config["max_iterations"] * 0.6),
        int(training_config["max_iterations"] * 0.8),
    )
    cfg.SOLVER.GAMMA = 0.5  # Reduce LR by 50% at each step
    cfg.SOLVER.WARMUP_ITERS = 200  # Warmup for stability
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.CHECKPOINT_PERIOD = training_config["checkpoint_period"]

    cfg.TEST.EVAL_PERIOD = training_config["eval_period"]

    cfg.OUTPUT_DIR = training_config["output_dir"]
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # Data augmentation for small dataset
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)

    return cfg


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 80)
    print("FASTER R-CNN FINE-TUNING ON RAIN DATASET")
    print("=" * 80)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    # Clear any existing registrations
    clear_datasets()

    # Create train/val split
    if (
        DATASET_CONFIG["val_images"] is None
        or DATASET_CONFIG["val_annotations"] is None
    ):
        print("Creating train/validation split...")
        train_json, val_json = create_train_val_split(
            DATASET_CONFIG["train_annotations"],
            DATASET_CONFIG["train_images"],
            val_ratio=DATASET_CONFIG["val_split_ratio"],
            output_dir="./temp_train_val_split",
        )
        train_images = DATASET_CONFIG["train_images"]
        val_images = DATASET_CONFIG["train_images"]
    else:
        train_json = DATASET_CONFIG["train_annotations"]
        val_json = DATASET_CONFIG["val_annotations"]
        train_images = DATASET_CONFIG["train_images"]
        val_images = DATASET_CONFIG["val_images"]
        print("Using provided train/validation split")

    # Register datasets
    print("\nRegistering datasets...")
    register_coco_instances("rain_train", {}, train_json, train_images)
    register_coco_instances("rain_val", {}, val_json, val_images)

    # Manually set thing_classes from the JSON
    with open(train_json, "r") as f:
        train_coco = json.load(f)
        categories = train_coco["categories"]
        thing_classes = [
            cat["name"] for cat in sorted(categories, key=lambda x: x["id"])
        ]

    MetadataCatalog.get("rain_train").thing_classes = thing_classes
    MetadataCatalog.get("rain_val").thing_classes = thing_classes

    train_classes = thing_classes
    print(f"  Classes detected: {train_classes}")
    print(f"  Number of classes: {len(train_classes)}")

    # Setup configuration
    cfg = setup_training_config(
        TRAINING_CONFIG, len(train_classes), TRAINING_CONFIG["device"]
    )

    print("\nTraining configuration:")
    print(f"  Base checkpoint: {cfg.MODEL.WEIGHTS}")
    print(f"  Output directory: {cfg.OUTPUT_DIR}")
    print(f"  Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    print(f"  Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"  Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"  Max iterations: {cfg.SOLVER.MAX_ITER}")

    # Create trainer and train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    # Save final model
    final_checkpoint = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    output_model_path = os.path.join("../FasterRCNN", FINETUNED_MODEL_NAME)

    if os.path.exists(final_checkpoint):
        shutil.copy(final_checkpoint, output_model_path)
        print(f"\nFinal model saved to: {output_model_path}")
        print(f"\nTo use this model in evaluation_a_rain.py:")
        print(f"  1. Set USE_FINETUNED_MODEL = True")
        print(f"  2. Run: python evaluation_a_rain.py")
    else:
        print(f"\nWarning: Final checkpoint not found at {final_checkpoint}")

    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("EVALUATING ON VALIDATION SET")
    print("=" * 80 + "\n")

    cfg.MODEL.WEIGHTS = final_checkpoint
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("rain_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "rain_val")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)

    print("\nValidation Results:")
    print("=" * 80)
    for key, value in results["bbox"].items():
        print(f"  {key}: {value:.4f}")
    print("=" * 80)

    print("\nFine-tuning complete! Model ready to use.")


if __name__ == "__main__":
    main()
