import os
import csv

IMAGE_DIR = "dataset/images"
MASK_DIR = "dataset/masks"
MASKED_DIR = "dataset/masked"
CSV_PATH = "dataset/dataset.csv"

# Collect all image IDs based on images folder
image_files = sorted(os.listdir(IMAGE_DIR))

rows = []

for img in image_files:
    img_id = os.path.splitext(img)[0]  # "000001"
    
    image_path = f"{IMAGE_DIR}/{img}"
    mask_path = f"{MASK_DIR}/{img_id}.png"
    masked_path = f"{MASKED_DIR}/{img}"
    
    # Check all files exist
    if os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(masked_path):
        rows.append([image_path, mask_path, masked_path])
    else:
        print("⚠ Missing file for:", img_id)

# Write CSV
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "mask", "masked"])
    writer.writerows(rows)

print("✔ CSV created successfully!")
print("Total samples written:", len(rows))
print("Saved at:", CSV_PATH)
