"""
Saves images from a DataFrame to disk as PNG files after resizing.
Args:
    df (pandas.DataFrame): DataFrame containing image data in a column named "image" as bytes.
    image_dir (Path or str): Directory path where the images will be saved.
    size (tuple, optional): Target size (width, height) for resizing images. Defaults to (224, 224).
Returns:
    list: List of filenames of the saved images.
Workflow:
    - Iterates over each row in the DataFrame with a progress bar.
    - Extracts image bytes from the "image" column (expects a dictionary with a "bytes" key).
    - Validates that the image data is in bytes format.
    - Opens the image using PIL, converts it to grayscale ("L" mode), and resizes it.
    - Saves the processed image as a PNG file in the specified directory, using a sequential filename.
    - Appends the filename to a list of saved images.
    - Handles and logs errors for rows with invalid or missing image data, skipping those rows.
    - Prints the total number of successfully saved images and returns the list of filenames.
"""
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def save_parquet_images(df, image_dir, size=(224, 224)):
    uuids = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Saving IU images"):
        try:
            img_data = row["image"].get("bytes", None)
            if img_data is None or not isinstance(img_data, bytes):
                raise ValueError("Image not in byte format.")

            img = Image.open(BytesIO(img_data)).convert("L")
            img = img.resize(size, Image.Resampling.LANCZOS)

            filename = f"iu_{i:04d}.png"
            img.save(image_dir / filename)
            uuids.append(filename)

        except Exception as e:
            print(f"Skipped row {i} due to error: {e}")
    
    print(f"Saved {len(uuids):,} images to: {image_dir}")
    return uuids