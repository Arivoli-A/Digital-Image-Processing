#!/usr/bin/env python3
"""
Batch inference helper to run CCNet on any folder of images.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# Add CCNet root to Python path so we can import modules
_SCRIPT_DIR = Path(__file__).resolve().parent
_CCNET_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_CCNET_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from networks.ccnet import Seg_Model

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CITYSCAPES_COLORS: List[Tuple[int, int, int]] = [
    (128, 64, 128),  # road
    (244, 35, 232),  # sidewalk
    (70, 70, 70),  # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170, 30),  # traffic light
    (220, 220, 0),  # traffic sign
    (107, 142, 35),  # vegetation
    (152, 251, 152),  # terrain
    (70, 130, 180),  # sky
    (220, 20, 60),  # person
    (255, 0, 0),  # rider
    (0, 0, 142),  # car
    (0, 0, 70),  # truck
    (0, 60, 100),  # bus
    (0, 80, 100),  # train
    (0, 0, 230),  # motorcycle
    (119, 11, 32),  # bicycle
]

CITYSCAPES_NAMES: List[str] = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CCNet on a folder of images.")
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Directory containing raw images for inference.",
    )
    parser.add_argument(
        "--weights",
        default="weights/ccnet_cityscapes_r1.pth",
        help="Path to the pretrained checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/custom",
        help="Where to write the predicted masks/overlays.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device string (cpu, cuda:0, mps, auto).",
    )
    parser.add_argument(
        "--long-side",
        type=int,
        default=1024,
        help="Resizes the longest image edge before inference (<=0 disables).",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=19,
        help="Number of segmentation classes.",
    )
    parser.add_argument(
        "--recurrence",
        type=int,
        default=1,
        help="Number of RCCA recurrence steps (1 for quickest).",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Also save color overlays blended with the RGB inputs.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.4,
        help="Opacity for overlay blending.",
    )
    parser.add_argument(
        "--save-labeled",
        action="store_true",
        help="Save images with text labels showing class names on detected regions.",
    )
    parser.add_argument(
        "--label-font-scale",
        type=float,
        default=0.6,
        help="Font scale for text labels (larger = bigger text).",
    )
    parser.add_argument(
        "--label-thickness",
        type=int,
        default=2,
        help="Thickness of text labels.",
    )
    parser.add_argument(
        "--label-mask",
        action="store_true",
        help="Also add labels to the mask image (colored segmentation).",
    )
    parser.add_argument(
        "--label-overlay",
        action="store_true",
        help="Also add labels to the overlay image (mask blended with original).",
    )
    parser.add_argument(
        "--min-region-size",
        type=int,
        default=500,
        help="Minimum pixel area for a region to be labeled.",
    )
    parser.add_argument(
        "--max-labels-per-class",
        type=int,
        default=3,
        help="Maximum number of labels per class (labels largest regions first).",
    )
    parser.add_argument(
        "--no-auto-font-scale",
        action="store_true",
        help="Disable automatic font scaling based on image size (use fixed font scale).",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def list_images(folder: Path) -> List[Path]:
    files = sorted(
        [p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTS and p.is_file()]
    )
    return files


def preprocess_image(image_path: Path, long_side: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], float]:
    raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise RuntimeError(f"Unable to read image: {image_path}")
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    original_size = raw.shape[:2]

    resized = raw.copy()
    scale = 1.0
    if long_side and long_side > 0:
        h, w = raw.shape[:2]
        max_side = max(h, w)
        if max_side != long_side:
            scale = min(long_side / float(max_side), 1.0)
            if scale != 1.0:
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                resized = cv2.resize(resized, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    image = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR).astype(np.float32)
    image -= IMG_MEAN
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, raw, original_size, scale


def build_palette(num_classes: int) -> Sequence[int]:
    palette = [0] * 256 * 3
    for idx, color in enumerate(CITYSCAPES_COLORS[:num_classes]):
        base = idx * 3
        palette[base : base + 3] = list(color)
    return palette


def colorize_mask(mask: np.ndarray, palette: Sequence[int]) -> Image.Image:
    img = Image.fromarray(mask, mode="P")
    img.putpalette(palette)
    return img


def add_labels_to_image(
    image: np.ndarray,
    mask: np.ndarray,
    class_names: List[str],
    font_scale: float = 0.6,
    thickness: int = 2,
    min_region_size: int = 500,
    max_labels_per_class: int = 3,
    auto_scale_font: bool = True,
) -> np.ndarray:
    """Add text labels to image showing class names for each unique semantic region.
    
    Only labels the largest connected components per class to avoid label clutter.
    """
    from scipy import ndimage
    
    labeled_image = image.copy()
    h, w = image.shape[:2]
    
    # Auto-scale font based on image size if enabled
    if auto_scale_font:
        # Base font scale for a 1024px wide image
        base_scale = 0.6
        # Scale based on image width (reference: 1024px)
        reference_width = 1024.0
        auto_font_scale = base_scale * (w / reference_width)
        # Clamp to reasonable bounds
        auto_font_scale = max(0.4, min(2.0, auto_font_scale))
        # Apply user's font_scale as a multiplier
        effective_font_scale = auto_font_scale * font_scale
        # Also scale thickness proportionally
        effective_thickness = max(1, int(thickness * (w / reference_width)))
    else:
        effective_font_scale = font_scale
        effective_thickness = thickness
    
    # Find unique classes in the mask
    unique_classes = np.unique(mask)
    
    # Track label positions to avoid overlaps
    label_positions = []
    
    for class_id in unique_classes:
        if class_id >= len(class_names):
            continue
        
        # Create binary mask for this class
        class_mask = (mask == class_id).astype(np.uint8)
        
        # Find connected components for this class
        labeled_array, num_features = ndimage.label(class_mask)
        
        # Get size of each component
        component_sizes = []
        for i in range(1, num_features + 1):
            size = np.sum(labeled_array == i)
            component_sizes.append((i, size))
        
        # Sort by size (largest first) and take top N
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        component_sizes = component_sizes[:max_labels_per_class]
        
        class_name = class_names[class_id]
        
        for component_id, size in component_sizes:
            # Skip small regions
            if size < min_region_size:
                continue
            
            # Get the component mask
            component_mask = (labeled_array == component_id).astype(np.uint8)
            
            # Find contours for this component
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            # Use the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate centroid for label placement
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, w_box, h_box = cv2.boundingRect(largest_contour)
                cx = x + w_box // 2
                cy = y + h_box // 2
            
            # Check if this position is too close to existing labels
            # Scale minimum distance with image size
            too_close = False
            min_distance = max(60, int(80 * (w / 1024.0)))  # Minimum pixel distance between labels
            for existing_x, existing_y in label_positions:
                distance = np.sqrt((cx - existing_x)**2 + (cy - existing_y)**2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(
                class_name, cv2.FONT_HERSHEY_SIMPLEX, effective_font_scale, effective_thickness
            )
            
            # Draw background rectangle for text (semi-transparent)
            # Scale padding with font size
            padding = max(4, int(6 * (effective_font_scale / 0.6)))
            overlay = labeled_image.copy()
            cv2.rectangle(
                overlay,
                (cx - text_w // 2 - padding, cy - text_h - baseline - padding),
                (cx + text_w // 2 + padding, cy + baseline + padding),
                (255, 255, 255),
                -1,
            )
            cv2.addWeighted(overlay, 0.8, labeled_image, 0.2, 0, labeled_image)
            
            # Draw text
            cv2.putText(
                labeled_image,
                class_name,
                (cx - text_w // 2, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                effective_font_scale,
                (0, 0, 0),
                effective_thickness,
                cv2.LINE_AA,
            )
            
            # Record this label position
            label_positions.append((cx, cy))
    
    return labeled_image


def run_inference() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise SystemExit(f"[run_inference] Image directory not found: {image_dir}")

    files = list_images(image_dir)
    if not files:
        raise SystemExit(f"[run_inference] No images found under {image_dir}")

    device = resolve_device(args.device)
    print(f"[run_inference] Using device: {device}")

    model = Seg_Model(num_classes=args.num_classes, recurrence=args.recurrence, pretrained_model=args.weights)
    model.to(device)
    model.eval()

    palette = build_palette(args.num_classes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image_path in tqdm(files, desc="Running inference", unit="img"):
            tensor, original_rgb, original_size, scale = preprocess_image(image_path, args.long_side)
            tensor = tensor.to(device)
            logits = model(tensor, None)
            if isinstance(logits, list):
                logits = logits[0]
            logits = F.interpolate(
                logits,
                size=tensor.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            if scale != 1.0:
                prediction = cv2.resize(
                    prediction,
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            color_mask = colorize_mask(prediction, palette)
            rel_path = image_path.relative_to(image_dir)
            stem = rel_path.stem
            mask_path = output_dir / f"{stem}_mask.png"
            
            class_names = CITYSCAPES_NAMES[:args.num_classes]
            
            # Save mask (optionally with labels)
            if args.label_mask:
                mask_rgb = np.array(color_mask.convert("RGB"))
                labeled_mask = add_labels_to_image(
                    mask_rgb,
                    prediction,
                    class_names,
                    font_scale=args.label_font_scale,
                    thickness=args.label_thickness,
                    min_region_size=args.min_region_size,
                    max_labels_per_class=args.max_labels_per_class,
                    auto_scale_font=not args.no_auto_font_scale,
                )
                Image.fromarray(labeled_mask).save(mask_path)
            else:
                color_mask.save(mask_path)

            if args.save_overlay:
                mask_rgb = color_mask.convert("RGB")
                overlay = (
                    args.overlay_alpha * np.array(mask_rgb) +
                    (1 - args.overlay_alpha) * original_rgb
                )
                overlay = overlay.astype(np.uint8)
                overlay_path = output_dir / f"{stem}_overlay.png"
                
                # Optionally add labels to overlay
                if args.label_overlay:
                    labeled_overlay = add_labels_to_image(
                        overlay,
                        prediction,
                        class_names,
                        font_scale=args.label_font_scale,
                        thickness=args.label_thickness,
                        min_region_size=args.min_region_size,
                        max_labels_per_class=args.max_labels_per_class,
                        auto_scale_font=not args.no_auto_font_scale,
                    )
                    Image.fromarray(labeled_overlay).save(overlay_path)
                else:
                    Image.fromarray(overlay).save(overlay_path)

            if args.save_labeled:
                # Create labeled version with text annotations on original image
                labeled_image = add_labels_to_image(
                    original_rgb,
                    prediction,
                    class_names,
                    font_scale=args.label_font_scale,
                    thickness=args.label_thickness,
                    min_region_size=args.min_region_size,
                    max_labels_per_class=args.max_labels_per_class,
                    auto_scale_font=not args.no_auto_font_scale,
                )
                labeled_path = output_dir / f"{stem}_labeled.png"
                Image.fromarray(labeled_image).save(labeled_path)

    print(f"[run_inference] Saved predictions to {output_dir}")


if __name__ == "__main__":
    run_inference()

