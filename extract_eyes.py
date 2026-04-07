import cv2
import os
import numpy as np
import pandas as pd

# ===== CONFIG =====
CSV_FILE = r"D:\major_phase2\archive (1)\list_landmarks_align_celeba.csv"
IMAGE_FOLDER = "sketches_512"   # Already 512x512 images
OUTPUT_FOLDER = "eye_only_output"

TARGET_SIZE = 512
ORIG_W = 178
ORIG_H = 218
# ==================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(CSV_FILE)

scale_x = TARGET_SIZE / ORIG_W
scale_y = TARGET_SIZE / ORIG_H

for _, row in df.iterrows():
    img_name = row["image_id"]
    img_path = os.path.join(IMAGE_FOLDER, img_name)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    # Scale coordinates
    lx = int(row["lefteye_x"] * scale_x)
    ly = int(row["lefteye_y"] * scale_y)

    rx = int(row["righteye_x"] * scale_x)
    ry = int(row["righteye_y"] * scale_y)

    # Estimate eye width dynamically
    inter_eye_distance = abs(rx - lx)
    eye_width = int(inter_eye_distance * 0.45)
    eye_height = int(eye_width * 0.5)   # smaller height avoids eyebrow

    for eye_type, (cx, cy) in [("left", (lx, ly)), ("right", (rx, ry))]:

        x1 = max(cx - eye_width // 2, 0)
        x2 = min(cx + eye_width // 2, TARGET_SIZE)

        y1 = max(cy - eye_height // 2, 0)
        y2 = min(cy + eye_height // 2, TARGET_SIZE)

        eye_crop = img[y1:y2, x1:x2]

        # Create white background
        white_bg = np.ones((512, 512, 3), dtype=np.uint8) * 255

        h, w = eye_crop.shape[:2]

        # Place eye at SAME original position
        paste_x1 = cx - w // 2
        paste_y1 = cy - h // 2

        white_bg[paste_y1:paste_y1+h, paste_x1:paste_x1+w] = eye_crop

        save_name = f"{os.path.splitext(img_name)[0]}_{eye_type}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, save_name), white_bg)

print("Eye-only extraction completed successfully.")