import os
import pandas as pd
import shutil
from tqdm import tqdm

# Paths
CSV_FILE = "final_soft_balanced.csv"
SOURCE_DIR = r"D:\Dataset\img_align_celeba\img_align_celeba"      # your original CelebA images
OUTPUT_DIR = "selected_images"

# Create target folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load list from CSV
df = pd.read_csv(CSV_FILE)
selected_files = df["image_id"].tolist()

print(f"🔍 Total selected images in CSV: {len(selected_files)}")
print("📂 Copying selected images...\n")

count = 0

for img_name in tqdm(selected_files):
    src = os.path.join(SOURCE_DIR, img_name)
    dst = os.path.join(OUTPUT_DIR, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)
        count += 1

print(f"\n✔ Done! Copied {count} images.")
print(f"📁 Saved in folder: {OUTPUT_DIR}")



