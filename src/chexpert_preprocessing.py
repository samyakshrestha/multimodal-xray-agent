
"""
This module provides utility functions for preprocessing X-ray images for the CheXpert dataset.
Functions:
-----------
_safe_mkdir(p: Path)
    Creates a directory at the given path in a thread-safe manner.
    Handles race conditions by catching the OSError if the directory already exists.
    Parameters:
        p (Path): The directory path to create.
process_one(img_path: Path)
    Processes a single image file:
        - Loads the image from the given path.
        - Converts the image to grayscale ("L" mode).
        - Applies a predefined transform (assumed to be defined elsewhere).
        - Adds a batch dimension and moves the tensor to the specified device (assumed to be defined elsewhere).
        - Converts the processed tensor back to a uint8 NumPy array.
        - Constructs a standardized output filename with a .png extension.
        - Saves the processed image as a PNG file in the output directory (assumed to be defined elsewhere).
    Returns:
        True if processing and saving succeed, False otherwise.
    Exceptions:
        Catches and prints any exception that occurs during processing, returning False.
"""
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