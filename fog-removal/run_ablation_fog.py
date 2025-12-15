import torch
import json
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Detectron2 & COCO
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Your Pipeline
from defog_pipeline import KimDefogPipeline

# --- CONFIGURATION ---
BASE_MODEL_PATH = "../FasterRCNN/untuned_model.pth"
ANNOTATIONS_PATH = (
    "./fog_dataset/labels.json"  # Use your specific test annotations if available
)
IMAGE_DIR = "./fog_dataset/input_images/unprocessed_images"
OUTPUT_DIR = "./ablation_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# The "Control" Configuration
DEFAULT_CONFIG = {
    # -- Hybrid Pipeline Params --
    "fusion_weight": 0.5,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (10, 10),
    "wavelet": "db4",
    "dwt_level": 1,
    "sharpening_factor": 0.7,
    # -- DarkChannelPrior Params --
    "patch_size": 15,
    "omega": 0.95,
    "guided_filter_radius": 60,
    "guided_filter_eps": 0.0001,
    "t_min": 0.1,
    "atm_percentile": 0.001,
}

# --- UTILS ---


def setup_predictor(weights_path):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
    cfg.MODEL.DEVICE = DEVICE
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.WEIGHTS = weights_path
    return DefaultPredictor(cfg)


def compute_metrics(coco_gt, predictions):
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0}

    # Suppress COCO prints
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    try:
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        results = {"AP": float(stats[0] * 100), "AP50": float(stats[1] * 100)}
    except Exception:
        results = {"AP": 0.0, "AP50": 0.0}

    sys.stdout = old_stdout
    return results


def run_experiment_eval(
    experiment_name, param_variations, predictor, coco_gt, image_ids
):
    print(f"\n--- Experiment: {experiment_name} ---")
    results = {}

    # Pre-calculate category mapping
    cat_name_to_id = {
        cat["name"]: cat["id"] for cat in coco_gt.loadCats(coco_gt.getCatIds())
    }
    class_idx_to_coco_id = {
        i: cat_name_to_id[c] for i, c in enumerate(CLASSES) if c in cat_name_to_id
    }
    pipeline_keys = {
        "fusion_weight",
        "clahe_clip_limit",
        "clahe_tile_grid_size",
        "wavelet",
        "dwt_level",
        "sharpening_factor",
    }

    for i, variation in enumerate(param_variations):
        # Create descriptive name
        var_name = "_".join([f"{k}-{v}" for k, v in variation.items()])
        if not var_name:
            var_name = "Default_Hybrid"

        # 1. Config Setup (Merge defaults with variation)
        config = DEFAULT_CONFIG.copy()
        config.update(variation)

        # Separate Pipeline args from DCP args
        # Extract 'mode' if it exists, so it doesn't get passed to the classes
        mode = config.pop("mode", "hybrid")
        pipeline_args = {k: v for k, v in config.items() if k in pipeline_keys}
        dcp_args = {k: v for k, v in config.items() if k not in pipeline_keys}

        pipeline = KimDefogPipeline(**pipeline_args, **dcp_args)

        # 2. Inference Loop
        predictions = []

        # Use a subset for faster debugging if needed, e.g., image_ids[:50]
        for img_id in tqdm(image_ids, desc=f"  Evaluating {var_name}", leave=False):
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(IMAGE_DIR, img_info["file_name"])

            raw_img = cv2.imread(img_path)
            if raw_img is None:
                continue

            # Dehaze
            if "mode" in variation and variation["mode"] == "baseline":
                processed_img = pipeline.dehaze(raw_img, return_baseline=True)
            else:
                processed_img = pipeline.dehaze(raw_img, return_baseline=False)

            # Predict (In-Memory)
            outputs = predictor(processed_img)
            instances = outputs["instances"]

            # Format Predictions
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

        # 3. Compute Metrics
        metrics = compute_metrics(coco_gt, predictions)
        results[var_name] = metrics
        print(
            f"    {var_name:<45} AP: {metrics['AP']:.2f} | AP50: {metrics['AP50']:.2f}"
        )

    return results


# --- MAIN ---
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading Model and Annotations...")
    predictor = setup_predictor(BASE_MODEL_PATH)

    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    coco_gt = COCO(ANNOTATIONS_PATH)
    sys.stdout = old_stdout

    image_ids = coco_gt.getImgIds()
    final_summary = {}

    # === Exp 1: Baseline vs Hybrid (Standard Config) ===
    # We establish the "Standard" performance here.
    exp1 = [{"mode": "baseline"}, {"mode": "hybrid"}]
    final_summary["1_Standard_Comparison"] = run_experiment_eval(
        "1_Standard_Comparison", exp1, predictor, coco_gt, image_ids
    )

    # === Exp 2: DWT Variations ===
    # The Baseline is identical to Exp 1, so we only run Hybrid variations.
    exp2 = []
    for w in ["db4", "haar"]:
        for l in [1, 2]:
            exp2.append({"wavelet": w, "dwt_level": l, "mode": "hybrid"})
    final_summary["2_DWT_Variations"] = run_experiment_eval(
        "2_DWT_Variations", exp2, predictor, coco_gt, image_ids
    )

    # === Exp 3: CLAHE Variations ===
    # The Baseline is identical to Exp 1, so we only run Hybrid variations.
    exp3 = [
        {"clahe_clip_limit": 1.0, "clahe_tile_grid_size": (8, 8), "mode": "hybrid"},
        {"clahe_clip_limit": 4.0, "clahe_tile_grid_size": (8, 8), "mode": "hybrid"},
        {"clahe_clip_limit": 2.0, "clahe_tile_grid_size": (16, 16), "mode": "hybrid"},
    ]
    final_summary["3_CLAHE_Variations"] = run_experiment_eval(
        "3_CLAHE_Variations", exp3, predictor, coco_gt, image_ids
    )

    # === Exp 4: DCP Internal Params (UPDATED) ===
    # Because these change the DCP itself, we run pairs: [Baseline, Hybrid] for every change.
    exp4 = []

    # 4a. Varying Patch Size
    for size in [5, 30]:
        exp4.append({"patch_size": size, "mode": "baseline"})  # See the new baseline
        exp4.append({"patch_size": size, "mode": "hybrid"})  # See the new hybrid

    # 4b. Varying Omega
    for om in [0.5, 0.8]:
        exp4.append({"omega": om, "mode": "baseline"})
        exp4.append({"omega": om, "mode": "hybrid"})

    final_summary["4_DCP_Params"] = run_experiment_eval(
        "4_DCP_Params", exp4, predictor, coco_gt, image_ids
    )

    with open(os.path.join(OUTPUT_DIR, "ablation_metrics.json"), "w") as f:
        json.dump(final_summary, f, indent=4)
    print(f"\nSaved all metrics to {os.path.join(OUTPUT_DIR, 'ablation_metrics.json')}")
