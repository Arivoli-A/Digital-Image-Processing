#!/usr/bin/env python3
"""
Example training script for Faster R-CNN on custom datasets.
This script demonstrates how to train on a COCO-format custom dataset.
"""
import argparse
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from register_custom_dataset import register_custom_coco_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on custom dataset")
    parser.add_argument(
        "--train-json",
        required=True,
        help="Path to training annotations JSON file (COCO format)",
    )
    parser.add_argument(
        "--train-images",
        required=True,
        help="Path to training images directory",
    )
    parser.add_argument(
        "--val-json",
        help="Path to validation annotations JSON file (COCO format)",
    )
    parser.add_argument(
        "--val-images",
        help="Path to validation images directory",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        required=True,
        help="Number of object classes (excluding background)",
    )
    parser.add_argument(
        "--thing-classes",
        nargs="+",
        help="List of class names (optional, for metadata)",
    )
    parser.add_argument(
        "--config-file",
        help="Path to config file (optional, uses default if not provided)",
    )
    parser.add_argument(
        "--weights",
        help="Path to pretrained weights (optional, downloads from model zoo if not provided)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Images per batch",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00025,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of training iterations",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    return parser.parse_args()


def setup_config(args):
    """Setup Detectron2 configuration."""
    cfg = get_cfg()
    
    # Load config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    else:
        # Use default Faster R-CNN config
        config_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
    
    # Set dataset paths
    cfg.DATASETS.TRAIN = (args.train_dataset_name,)
    if args.val_json:
        cfg.DATASETS.TEST = (args.val_dataset_name,)
    
    # Set number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    
    # Set model weights
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    else:
        # Use pretrained weights from model zoo
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
    
    # Training hyperparameters
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (int(args.max_iter * 0.6), int(args.max_iter * 0.8))
    
    # Output directory
    cfg.OUTPUT_DIR = args.output_dir
    
    # Other settings
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    
    return cfg


def main():
    args = parse_args()
    
    # Register training dataset
    args.train_dataset_name = "custom_train"
    register_custom_coco_dataset(
        name=args.train_dataset_name,
        json_file=args.train_json,
        image_root=args.train_images,
        thing_classes=args.thing_classes,
    )
    print(f"Registered training dataset: {args.train_dataset_name}")
    
    # Register validation dataset if provided
    if args.val_json and args.val_images:
        args.val_dataset_name = "custom_val"
        register_custom_coco_dataset(
            name=args.val_dataset_name,
            json_file=args.val_json,
            image_root=args.val_images,
            thing_classes=args.thing_classes,
        )
        print(f"Registered validation dataset: {args.val_dataset_name}")
    
    # Setup configuration
    cfg = setup_config(args)
    
    # Create output directory
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = DefaultTrainer(cfg)
    
    # Resume or start training
    trainer.resume_or_load(resume=args.resume)
    
    print(f"\nStarting training...")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    print(f"Number of classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print()
    
    # Train
    trainer.train()
    
    print(f"\nTraining complete! Checkpoints saved to {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()


