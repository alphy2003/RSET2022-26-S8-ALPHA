import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def classify_face_shape(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "unknown"
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return "unknown"

        landmarks = results.multi_face_landmarks[0]

        h, w, _ = img.shape

        # Example landmark indices
        left_jaw = landmarks.landmark[234]
        right_jaw = landmarks.landmark[454]
        chin = landmarks.landmark[152]
        forehead = landmarks.landmark[10]

        jaw_width = abs((right_jaw.x - left_jaw.x) * w)
        face_height = abs((chin.y - forehead.y) * h)

        ratio = face_height / jaw_width

        if ratio < 1.1:
            return "round"
        elif 1.1 <= ratio <= 1.3:
            return "square"
        else:
            return "oval"

# Load CSV
df = pd.read_csv("frontal_images.csv")

image_folder = r"D:\Dataset\img_align_celeba\img_align_celeba"

face_shapes = []

for img_id in df['image_id']:
    img_path = os.path.join(image_folder, img_id)
    shape = classify_face_shape(img_path)
    face_shapes.append(shape)

df['face_shape'] = face_shapes

df.to_csv("categorized_faces.csv", index=False)

print("New CSV created successfully!")