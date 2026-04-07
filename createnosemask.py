

import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Paths
csv_path = "nose_dataset_cleaned.csv"          # your CSV file
image_folder = r"D:\Dataset\img_align_celeba\img_align_celeba"         # dataset folder
output_folder = "nose_masks"   # masks output
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(csv_path)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Nose landmarks
NOSE_POINTS = [1, 2, 5, 4, 98, 327, 168, 197, 195, 6]

for img_id in df["image_id"]:

    img_path = os.path.join(image_folder, str(img_id))

    if not os.path.exists(img_path):
        img_path += ".jpg"
    if not os.path.exists(img_path):
        img_path += ".png"

    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        continue

    landmarks = results.multi_face_landmarks[0]

    points = []
    for idx in NOSE_POINTS:
        lm = landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)

    # Create black mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Fill nose polygon
    cv2.fillPoly(mask, [points], 255)

    # --------- ENLARGE AREA ---------

    # 1) Dilate (expand region)
    kernel = np.ones((25,25), np.uint8)  # increase size here
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 2) Smooth edges
    mask = cv2.GaussianBlur(mask, (9,9), 0)

    # Save mask
    save_path = os.path.join(output_folder, str(img_id) + ".png")
    cv2.imwrite(save_path, mask)

    print("Saved:", img_id)

print("Done!")
