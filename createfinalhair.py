import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

def finalize_training_data(sketch_dir, mask_dir, output_dir, csv_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_path)
    image_ids = df['image_id'].tolist()

    print("Finalizing Hair-Only sketches for training...")

    for img_id in tqdm(image_ids):
        sketch_path = os.path.join(sketch_dir, img_id)
        # Masks are usually .png to stay sharp
        mask_path = os.path.join(mask_dir, img_id.replace('.jpg', '.png')) 
        
        sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if sketch is None or mask is None:
            continue

        # 1. Invert the sketch (Black lines become White)
        inv_sketch = 255 - sketch

        # 2. Apply Mask (Keeps only white lines inside the mask area)
        hair_only_inv = cv2.bitwise_and(inv_sketch, inv_sketch, mask=mask)

        # 3. Invert back (Lines become Black, Background becomes White)
        final_hair_data = 255 - hair_only_inv

        # 4. Save to your training folder
        cv2.imwrite(os.path.join(output_dir, img_id), final_hair_data)

if __name__ == "__main__":
    SKETCH_FOLDER = r"D:\major_phase2\pro_sketches"
    MASK_FOLDER = r"D:\major_phase2\hair_masks"
    TRAIN_DATA_FOLDER = r"D:\major_phase2\hair_training_set"
    CSV_PATH = r"D:\major_phase2\front_facing_hair_dataset.csv"

    finalize_training_data(SKETCH_FOLDER, MASK_FOLDER, TRAIN_DATA_FOLDER, CSV_PATH)