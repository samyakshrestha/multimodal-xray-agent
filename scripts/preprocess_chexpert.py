# Preprocess CheXpert dataset images to 224x224 resolution

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm 

def preprocess_chexpert(in_dir, out_dir, target_size=(224, 224)):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for root, _, files in os.walk(in_dir):
        for fname in files:
            if fname.lower().endswith(".jpg"):
                fpath = Path(root) / fname
                rel_parts = fpath.relative_to(in_dir).parts  # e.g., ['train', 'patient00001', 'study1', 'view1_frontal.jpg']
                if len(rel_parts) >= 3:
                    patient, study, view = rel_parts[-3:]
                    out_name = f"{patient}_{study}_{fname}"
                else:
                    out_name = fname

                try:
                    img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    img = cv2.resize(img, target_size)
                    out_path = out_dir / out_name
                    cv2.imwrite(str(out_path), img)
                    count += 1
                except Exception as e:
                    print(f"[ERROR] Failed on: {fpath} — {e}")
    
    print(f"\nPreprocessing complete: {count} images written to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="Path to chexpert_raw/train/")
    parser.add_argument("--out_dir", type=str, required=True, help="Output path for resized 224×224 images")
    args = parser.parse_args()

    preprocess_chexpert(args.in_dir, args.out_dir) 