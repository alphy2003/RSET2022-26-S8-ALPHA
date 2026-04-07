import pandas as pd
import shutil
import os

# --- CONFIGURATION ---
# 1. Path to your cleaned CSV (e.g., the 11k balanced one or your training CSV)
CSV_PATH = r"D:\major_phase2\final_forensic_manifest.csv"

# 2. Path where ALL your original images are stored
SOURCE_DIR = r"D:\Dataset\img_align_celeba\img_align_celeba"

# 3. Where you want to put the copied images
TARGET_DIR = r"D:\major_phase2\eye_images_12k"

# Create the target folder if it doesn't exist
os.makedirs(TARGET_DIR, exist_ok=True)

def copy_dataset_images():
    print(f"Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Assuming your CSV has a column named 'image_id'
    image_list = df['image_id'].tolist()
    total_images = len(image_list)
    
    print(f"Found {total_images} filenames in CSV. Starting copy process...")

    copied_count = 0
    missing_count = 0

    for i, img_name in enumerate(image_list):
        source_path = os.path.join(SOURCE_DIR, img_name)
        target_path = os.path.join(TARGET_DIR, img_name)
        
        if os.path.exists(source_path):
            # shutil.copy2 preserves metadata like timestamps
            shutil.copy2(source_path, target_path)
            copied_count += 1
        else:
            missing_count += 1
            if missing_count <= 5: # Print only the first few missing
                print(f"Warning: {img_name} not found in source directory.")

        # Print progress every 1000 images
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i + 1}/{total_images} processed...")

    print("\n--- Process Complete ---")
    print(f"Successfully copied: {copied_count}")
    print(f"Missing images:      {missing_count}")
    print(f"Destination:         {TARGET_DIR}")

if __name__ == "__main__":
    copy_dataset_images()