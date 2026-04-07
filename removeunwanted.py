import os
import shutil

# === CHANGE THESE PATHS ===
IMAGE_DIR = r"D:\Dataset\img_align_celeba\img_align_celeba"          # Folder with all original CelebA images
MISSING_FILE = r"D:\CelebA\missing_eye_masks.txt"  # The txt file containing missing image names (no mask)
DEST_DIR = r"D:\CelebA\Missing_Images" # Folder where missing images will be copied

# Create destination folder if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Read missing file names
with open(MISSING_FILE, "r") as f:
    missing_images = [line.strip() for line in f if line.strip()]

count = 0

for img_name in missing_images:
    # Ensure proper extension (CelebA images are usually .jpg)
    if not img_name.lower().endswith(".jpg"):
        img_name = img_name + ".jpg"
    
    src_path = os.path.join(IMAGE_DIR, img_name)
    dst_path = os.path.join(DEST_DIR, img_name)

    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        count += 1
    else:
        print(f"❌ Image not found: {img_name}")

print(f"\n-----------------------------")
print(f"Copied: {count} missing images")
print(f"Saved to: {DEST_DIR}")
print(f"-----------------------------")
