# This script preprocesses CheXpert images on a GPU, resizing them to 224x224 pixels and converting them to grayscale.

import os
import argparse 
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from torchvision import transforms

# Global resize transform (GPU)
transform = transforms.Compose([ 
    transforms.Resize((224, 224)), # Resize to 224x224
    transforms.ToTensor(),  # [0,1], float32
])

# Function to process a single image
def process_image(in_path, out_path):
    try: 
        img = Image.open(in_path).convert("L")  # Grayscale
        tensor = transform(img).unsqueeze(0).to("cuda")  # [1, 1, 224, 224]
        img_resized = (tensor.squeeze().cpu().numpy() * 255).astype("uint8") # Convert to uint8 for saving
        Image.fromarray(img_resized).save(out_path) # Save resized image
        return True
    except Exception as e:
        print(f"[ERROR] {in_path}: {e}")
        return False

# Function to preprocess CheXpert dataset images using GPU
def preprocess_chexpert_gpu(in_dir, out_dir, max_workers=4):
    in_dir = Path(in_dir) # Ensure in_dir is a Path object
    out_dir = Path(out_dir) # Ensure out_dir is a Path object
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = []
    for root, _, files in os.walk(in_dir): # Walk through all subdirectories
        for f in files:
            if f.lower().endswith(".jpg"): 
                full_path = Path(root) / f 
                rel_parts = full_path.relative_to(in_dir).parts
                if len(rel_parts) >= 3: 
                    patient, study, fname = rel_parts[-3:] # e.g., ['train', 'patient00001', 'study1', 'view1_frontal.jpg']
                    out_name = f"{patient}_{study}_{fname}" 
                    out_name = f.name
                out_path = out_dir / out_name
                all_paths.append((full_path, out_path))
    with ThreadPoolExecutor(max_workers=max_workers) as executor: # Create a thread pool for parallel processing
    # Use tqdm to show progress bar
        results = list(tqdm(executor.map(lambda args: process_image(*args), all_paths), total=len(all_paths)))

    success = sum(results) # Count successful image writes
    print(f"\nDone: {success} / {len(all_paths)} images written to {out_dir}")

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True, help="Path to chexpert_raw/train/")
    parser.add_argument("--out_dir", type=str, required=True, help="Output path for resized images")
    parser.add_argument("--workers", type=int, default=4, help="Max number of CPU threads for I/O")
    args = parser.parse_args()

    torch.set_default_device("cuda")  # force torch ops on GPU
    preprocess_chexpert_gpu(args.in_dir, args.out_dir, args.workers)