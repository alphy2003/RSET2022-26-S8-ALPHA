import os

IMAGE_DIR = r"D:\Dataset\img_align_celeba\img_align_celeba"   # folder with .jpg images
MASK_DIR = r"D:\CelebA\EyeMasks"  # folder with *_eye_mask filenames
OUT_FILE = r"D:\CelebA\missing_eye_masks.txt"

# Get image names without extension
images = sorted([os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")])

# Extract numeric ID from mask filenames (before `_eye_mask`)
masks = set()
for f in os.listdir(MASK_DIR):
    if "_eye_mask" in f:
        base_name = f.split("_eye_mask")[0]     # take only `000001`
        masks.add(base_name)

# Compare
missing = [img for img in images if img not in masks]

# Save results
with open(OUT_FILE, "w") as f:
    for name in missing:
        f.write(name + "\n")

# Summary
print("\n----- SUMMARY -----")
print(f"Total images: {len(images)}")
print(f"Masks found: {len(masks)}")
print(f"Missing masks: {len(missing)}")
print(f"Missing list saved to: {OUT_FILE}")
print("-------------------\n")
