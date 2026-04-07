import os
import pandas as pd
import shutil  # This is the correct library for copying files

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_PATH = r"D:\major_phase2\train.csv"
SOURCE_DIR = r"D:\Dataset\img_align_celeba\img_align_celeba"
DEST_DIR = r"D:\major_phase2\subset_images"

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)
    print(f"Created folder: {DEST_DIR}")

# ==========================================
# 2. FILE COPYING PROCESS
# ==========================================
def run_transfer():
    df = pd.read_csv(CSV_PATH)
    image_list = df['image_id'].tolist()
    
    total = len(image_list)
    print(f"Total images to copy: {total}")
    
    success = 0
    fail = 0

    for i, img_name in enumerate(image_list):
        source = os.path.join(SOURCE_DIR, img_name)
        dest = os.path.join(DEST_DIR, img_name)
        
        if os.path.exists(source):
            shutil.copy2(source, dest)
            success += 1
        else:
            fail += 1
        
        # Simple progress printer
        if i % 1000 == 0:
            print(f"Progress: {i}/{total} images processed...")

    print(f"\nDone! Success: {success} | Failed: {fail}")

if __name__ == "__main__":
    run_transfer()