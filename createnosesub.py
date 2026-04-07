import os
import pandas as pd
import shutil
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
# The CSV created by your mask generation script
CSV_PATH = r"D:\major_phase2\nose_final_train.csv"

# Where all 202,599 CelebA images are currently stored
MASTER_IMG_DIR = r"D:\Dataset\img_align_celeba\img_align_celeba"

# The new folder for your nose training images
DEST_DIR = r"D:\major_phase2\nose_subset_images"

# Create folder if it doesn't exist
if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)
    print(f"Created folder: {DEST_DIR}")

# ==========================================
# 2. SUBSET CREATION PROCESS
# ==========================================
def create_subset():
    # Load the list of valid images (those that have masks)
    df = pd.read_csv(CSV_PATH)
    image_list = df['image_id'].tolist()
    
    total_to_copy = len(image_list)
    print(f"Starting copy of {total_to_copy} images...")
    
    success = 0
    fail = 0

    # Using tqdm for a progress bar
    for img_name in tqdm(image_list, desc="Copying Images"):
        source_path = os.path.join(MASTER_IMG_DIR, img_name)
        dest_path = os.path.join(DEST_DIR, img_name)
        
        if os.path.exists(source_path):
            # copy2 preserves metadata like timestamps
            shutil.copy2(source_path, dest_path)
            success += 1
        else:
            fail += 1

    print(f"\n--- Transfer Complete ---")
    print(f"Successfully moved: {success} images")
    print(f"Failed (not found in master): {fail} images")
    print(f"Your nose training subset is ready in: {DEST_DIR}")

if __name__ == "__main__":
    create_subset()