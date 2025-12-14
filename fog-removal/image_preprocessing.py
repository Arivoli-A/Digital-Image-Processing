from defog_pipeline import KimDefogPipeline
from dcp import DarkChannelPrior
from pathlib import Path
import cv2
import os

if __name__ == "__main__":
    # Setup image paths
    base_dir = Path("./fog_dataset/input_images")
    image_dir = base_dir / "unprocessed_images"
    kim_processed_image_dir = base_dir / "kim_pipeline_images"
    dcp_processed_image_dir = base_dir / "dcp_images"

    # Create directories
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # Image processing pipelines
    kim_dehazer = KimDefogPipeline()
    dcp_dehazer = DarkChannelPrior()

    for i, path in enumerate(image_dir.glob("*.png")):
        foggy_img = cv2.imread(path)
        kim_dehazed_img = kim_dehazer.dehaze(foggy_img)
        dcp_dehazed_img = dcp_dehazer.dehaze(foggy_img)

        # Paths
        kim_processed_path = kim_processed_image_dir / f"{path.stem}.png"
        dcp_processed_path = dcp_processed_image_dir / f"{path.stem}.png"

        # Write images
        cv2.imwrite(kim_processed_path, kim_dehazed_img)
        cv2.imwrite(dcp_processed_path, dcp_dehazed_img)
