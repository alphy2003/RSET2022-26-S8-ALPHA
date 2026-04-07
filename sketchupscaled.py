import cv2
import os
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FOLDER = r"D:\Dataset\img_align_celeba\img_align_celeba"
OUTPUT_FOLDER = 'sketches_512'

TARGET_SIZE = (512, 512)   # 👈 Upscale size

BLUR_KSIZE = 45            # Must be odd number (controls thickness)
DARKNESS_FACTOR = 2      # >1.0 makes lines darker

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created directory: {OUTPUT_FOLDER}")

def to_artistic_sketch(img, blur_ksize=45, darkness_factor=1.0):
    """
    Converts image to dark artistic sketch and resizes to 512x512.
    """

    # 0️⃣ Resize FIRST (important for consistent blur thickness)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

    # 1️⃣ Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2️⃣ Invert grayscale
    inverted_gray = 255 - gray

    # 3️⃣ Gaussian Blur
    blurred = cv2.GaussianBlur(inverted_gray, (blur_ksize, blur_ksize), 0)

    # 4️⃣ Invert blurred image
    inverted_blurred = 255 - blurred

    # 5️⃣ Color Dodge blend
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)

    # 6️⃣ Darken using Gamma Correction
    sketch_float = sketch / 255.0
    sketch_darkened = np.power(sketch_float, darkness_factor)
    sketch_final = (sketch_darkened * 255).astype(np.uint8)

    return sketch_final


# --- MAIN PROCESSING LOOP ---
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

if not os.path.exists(INPUT_FOLDER):
    print(f"Error: Input folder not found at {INPUT_FOLDER}")
else:
    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]
    print(f"Found {len(all_files)} images to process.")

    for filename in tqdm(all_files, desc="Converting to 512x512 Dark Sketches"):
        try:
            img_path = os.path.join(INPUT_FOLDER, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            sketch = to_artistic_sketch(
                img,
                blur_ksize=BLUR_KSIZE,
                darkness_factor=DARKNESS_FACTOR
            )

            save_path = os.path.join(OUTPUT_FOLDER, filename)
            cv2.imwrite(save_path, sketch)

        except Exception as e:
            print(f"Error on {filename}: {e}")

    print("\nDone! 512x512 Dark Sketches saved in:", OUTPUT_FOLDER)