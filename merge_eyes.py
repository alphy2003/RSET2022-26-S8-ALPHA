import cv2
import os
import numpy as np

# ===== CONFIG =====
INPUT_FOLDER = "eye_only_output"
OUTPUT_FOLDER = "merged_eyes"
# ==================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

all_files = os.listdir(INPUT_FOLDER)

# Get only left eye files
left_files = [f for f in all_files if "_left" in f]

for left_file in left_files:

    base_name = left_file.replace("_left", "")
    right_file = base_name.replace(".jpg", "_right.jpg")

    left_path = os.path.join(INPUT_FOLDER, left_file)
    right_path = os.path.join(INPUT_FOLDER, right_file)

    if not os.path.exists(right_path):
        continue

    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    if left_img is None or right_img is None:
        continue

    # Start with white background
    merged = np.ones((512, 512, 3), dtype=np.uint8) * 255

    # Mask where pixel is not white
    left_mask = np.any(left_img < 250, axis=2)
    right_mask = np.any(right_img < 250, axis=2)

    # Overlay left eye
    merged[left_mask] = left_img[left_mask]

    # Overlay right eye
    merged[right_mask] = right_img[right_mask]

    save_name = base_name.replace(".jpg", "_merged.jpg")
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, save_name), merged)

print("All eyes merged successfully.")