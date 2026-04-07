import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_face_mesh = mp.solutions.face_mesh

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

output_folder = "face_outlines"
os.makedirs(output_folder, exist_ok=True)

def create_clean_outline(image_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return False

        landmarks = results.multi_face_landmarks[0]

        # White background
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

        points = []
        for idx in FACE_OVAL:
            x = int(landmarks.landmark[idx].x * w)
            y = int(landmarks.landmark[idx].y * h)
            points.append([x, y])

        points = np.array(points, np.int32)

        # Draw smooth closed outline
        cv2.polylines(canvas, [points], isClosed=True, color=(0, 0, 0), thickness=3)

        cv2.imwrite(save_path, canvas)
        return True


# Load CSV
df = pd.read_csv("frontal_images.csv")
image_folder = r"D:\Dataset\img_align_celeba\img_align_celeba"

outline_paths = []

for img_id in df['image_id']:
    img_path = os.path.join(image_folder, img_id)
    save_path = os.path.join(output_folder, img_id)

    if create_clean_outline(img_path, save_path):
        outline_paths.append(save_path)
    else:
        outline_paths.append("failed")

df["outline_image"] = outline_paths
df.to_csv("outline_faces.csv", index=False)

print("Exact face outlines generated successfully!")