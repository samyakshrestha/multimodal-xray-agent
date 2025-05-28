# Preprocess CheXpert dataset images to 224x224 resolution

import errno
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

def _safe_mkdir(p: Path):
    """Thread-safe mkdir that handles race conditions."""
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def process_one(img_path: Path, out_dir: Path, transform, device="cpu"):
    """
    Preprocess a single X-ray image:
    - Converts to grayscale
    - Resizes and normalizes
    - Saves as .png to out_dir
    """
    try:
        img = Image.open(img_path).convert("L")                      # Convert to grayscale
        tensor = transform(img).unsqueeze(0).to(device)              # Resize + send to device
        arr = (tensor.squeeze().cpu().numpy() * 255).astype("uint8")  # Convert to uint8 image

        fname = img_path.stem + ".png"                               # Standardize filename
        out_path = out_dir / fname
        _safe_mkdir(out_path.parent)
        Image.fromarray(arr).save(out_path, format="PNG")            # Save as PNG
        return True
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")
        return False