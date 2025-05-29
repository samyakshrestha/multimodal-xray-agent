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
    
def process_one(img_path: Path):
    try:
        # Load, convert to grayscale, apply transform, move to GPU
        img = Image.open(img_path).convert("L")
        tensor = transform(img).unsqueeze(0).to(device)

        # Convert back to uint8 NumPy array
        arr = (tensor.squeeze().cpu().numpy() * 255).astype("uint8")

        # Build .png filename (standardizing format)
        fname = img_path.stem + ".png"  # Removes .jpg/.jpeg/etc., replaces with .png
        out_path = OUT_DIR / fname

        # Save as PNG
        Image.fromarray(arr).save(out_path, format="PNG")
        return True
    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")
        return False