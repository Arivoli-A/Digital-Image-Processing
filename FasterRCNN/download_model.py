#!/usr/bin/env python3
"""
Helper script to download Detectron2 model weights manually.
This can help if the automatic download gets stuck.
"""
import os
import ssl
import sys
from pathlib import Path

# Fix SSL certificate issues
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
    ssl._create_default_https_context = ssl._create_unverified_context
except ImportError:
    pass

from detectron2 import model_zoo
from detectron2.config import get_cfg

def download_model(model_name="faster_rcnn_R_50_FPN_3x"):
    """Download model weights manually."""
    print(f"Downloading model: {model_name}")
    
    try:
        cfg = get_cfg()
        config_path = f"COCO-Detection/{model_name}.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        
        print(f"Config loaded: {config_path}")
        print(f"Model weights URL: {model_zoo.get_checkpoint_url(f'COCO-Detection/{model_name}.yaml')}")
        
        # This will trigger the download
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{model_name}.yaml")
        print(f"\nModel weights will be cached at: ~/.cache/detectron2/")
        print("Download started... This may take a few minutes.")
        
        # Try to actually download by accessing it
        import urllib.request
        url = cfg.MODEL.WEIGHTS
        print(f"\nDownloading from: {url}")
        
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            bar_length = 40
            filled = int(bar_length * downloaded / total_size)
            bar = '=' * filled + '-' * (bar_length - filled)
            sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)')
            sys.stdout.flush()
        
        filename = url.split('/')[-1]
        cache_dir = Path.home() / ".cache" / "detectron2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / filename
        
        if output_path.exists():
            print(f"\n✓ Model already downloaded at: {output_path}")
            print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            print(f"\nDownloading to: {output_path}")
            urllib.request.urlretrieve(url, output_path, show_progress)
            print(f"\n✓ Download complete!")
            print(f"  Saved to: {output_path}")
            print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "faster_rcnn_R_50_FPN_3x"
    download_model(model)


