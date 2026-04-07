import cv2
import os
import numpy as np

INPUT_FOLDER = r"D:\major_phase2\hair_only"
OUTPUT_FOLDER = r"D:\major_phase2\hair_sketch"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def hair_black_white(img):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OTSU threshold (auto best threshold)
    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Invert so hair becomes black
    mask = 255 - mask

    return mask

for file in os.listdir(INPUT_FOLDER):

    if file.lower().endswith((".jpg",".png",".jpeg")):

        path = os.path.join(INPUT_FOLDER, file)
        img = cv2.imread(path)

        if img is None:
            continue

        result = hair_black_white(img)

        cv2.imwrite(
            os.path.join(OUTPUT_FOLDER, file),
            result
        )

print("✅ Done! Hair = Black, Background = White")
