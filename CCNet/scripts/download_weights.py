#!/usr/bin/env python3
"""
Utility to fetch pretrained CCNet checkpoints from Google Drive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import gdown  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "Missing dependency 'gdown'. Install it with:\n"
        "  pip install gdown"
    ) from exc


MODEL_INDEX = {
    "r1": {
        "file_id": "13j06I4e50T41j_2HQl4sksrLZihax94L",
        "filename": "ccnet_cityscapes_r1.pth",
        "description": "R=1 checkpoint (~77.9 mIoU on Cityscapes val)",
    },
    "r2": {
        "file_id": "1IxXm8qxKmfDPVRtT8uuDNEvSQsNVTfLC",
        "filename": "ccnet_cityscapes_r2.pth",
        "description": "R=2 checkpoint (~79.7 mIoU on Cityscapes val)",
    },
    "r2_ohem": {
        "file_id": "1eiX1Xf1o16DvQc3lkFRi4-Dk7IBVspUQ",
        "filename": "ccnet_cityscapes_r2_ohem.pth",
        "description": "R=2 + OHEM checkpoint (~80.0 mIoU on Cityscapes val)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CCNet pretrained weights into the repo."
    )
    parser.add_argument(
        "--model",
        choices=MODEL_INDEX.keys(),
        default="r1",
        help="Checkpoint variant to fetch.",
    )
    parser.add_argument(
        "--output-dir",
        default="weights",
        help="Destination directory relative to the repo root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = MODEL_INDEX[args.model]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / info["filename"]

    if output_file.exists() and not args.force:
        print(f"[download_weights] Reusing existing file: {output_file}")
        return

    url = f"https://drive.google.com/uc?id={info['file_id']}"
    print(f"[download_weights] Downloading {args.model} -> {output_file}")
    gdown.download(url, str(output_file), quiet=False)
    print("[download_weights] Done.")


if __name__ == "__main__":
    main()

