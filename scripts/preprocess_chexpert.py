# Preprocess CheXpert dataset images to 224x224 resolution

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm 

def preprocess_chexpert(in_dir, out_dir, target_size=(224, 224)):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist

    count = 0
    for root, _, files in os.walk(in_dir):  # Walk through all files in input directory
        for fname in files:
            if fname.lower().endswith(".jpg"):  # Process only .jpg files
                fpath = Path(root) / fname
                rel_parts = fpath.relative_to(in_dir).parts  # Get relative path parts
                # Construct output filename using patient, study, and original filename if possible
                if len(rel_parts) >= 3:
                    patient, study, view = rel_parts[-3:]
                    out_name = f"{patient}_{study}_{fname}"
                else:
                    out_name = fname

                try:
                    img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
                    if img is None:
                        continue  # Skip if image can't be read
                    img = cv2.resize(img, target_size)  # Resize image to target size
                    out_path = out_dir / out_name
                    cv2.imwrite(str(out_path), img)  # Save preprocessed image
                    count += 1
                except Exception as e:
                    print(f"[ERROR] Failed on: {fpath} — {e}")  # Print error if processing fails
    
    print(f"\nPreprocessing complete: {count} images written to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="Path to chexpert_raw/train/")
    parser.add_argument("--out_dir", type=str, required=True, help="Output path for resized 224×224 images")
    args = parser.parse_args()

    preprocess_chexpert(args.in_dir, args.out_dir)  # Run preprocessing with provided arguments