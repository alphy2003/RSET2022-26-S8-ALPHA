import cv2
import os
import numpy as np
from tqdm import tqdm

INPUT_DIR = r"D:\major_phase2\nose_extracted"
OUTPUT_DIR = r"D:\major_phase2\nose_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for img_name in tqdm(os.listdir(INPUT_DIR)):

    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create mask (detect nose area)
    _, mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # Clean small noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Create white background
    result = np.ones_like(gray) * 255

    # Copy only nose region
    result[mask == 255] = gray[mask == 255]

    # Save
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), result)

print("Done! Clean noses saved.")
