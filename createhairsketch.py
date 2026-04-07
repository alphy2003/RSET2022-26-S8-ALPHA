import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def create_manual_artistic_sketch(img_info):
    img_id, input_path, output_path = img_info
    
    if os.path.exists(output_path):
        return

    img = cv2.imread(input_path)
    if img is None:
        return

    try:
        # 1. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Invert the image
        inverted_gray = 255 - gray

        # 3. Apply a heavy Gaussian Blur to the inverted image
        # The blur size (21, 21) determines the thickness of the lines
        blurred_img = cv2.GaussianBlur(inverted_gray, (21, 21), 0)

        # 4. Color Dodge Blend (The "Magic" Step)
        # This blends the original gray with the blurred inverted image
        # It creates sharp black lines on a white background
        sketch = cv2.divide(gray, 255 - blurred_img, scale=256)

        # 5. Increase Contrast (Make the black lines darker)
        # We use a threshold to "punch up" the black lines
        _, final_sketch = cv2.threshold(sketch, 240, 255, cv2.THRESH_BINARY)

        # 6. Denoise
        final_sketch = cv2.medianBlur(final_sketch, 3)

        cv2.imwrite(output_path, final_sketch)
        
    except Exception as e:
        print(f"Error on {img_id}: {e}")

def main(csv_path, img_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    image_ids = df['image_id'].tolist()

    print(f"Generating Manual Artistic Sketches for {len(image_ids)} images...")

    tasks = []
    for img_id in image_ids:
        tasks.append((img_id, os.path.join(img_dir, img_id), os.path.join(save_dir, img_id)))

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(create_manual_artistic_sketch, tasks), total=len(tasks)))

if __name__ == "__main__":
    INPUT_CSV = r"D:\major_phase2\front_facing_hair_dataset.csv"
    IMAGE_FOLDER = r"D:\Dataset\img_align_celeba\img_align_celeba"
    SKETCH_SAVE_DIR = r"D:\major_phase2\pro_sketches"

    main(INPUT_CSV, IMAGE_FOLDER, SKETCH_SAVE_DIR)