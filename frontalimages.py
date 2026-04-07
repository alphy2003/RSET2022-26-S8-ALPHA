import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

INPUT_FOLDER = r"D:\Dataset\img_align_celeba\img_align_celeba"
CSV_OUTPUT = "frontal_images.csv"

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

# 3D model points (generic face)
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose
    (-30.0, -30.0, -30.0),  # Left eye
    (30.0, -30.0, -30.0),   # Right eye
    (-40.0, 30.0, -30.0),   # Left mouth
    (40.0, 30.0, -30.0)     # Right mouth
])

def is_frontal(img):
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return False

    lm = results.multi_face_landmarks[0].landmark

    image_points = np.array([
        (lm[1].x*w, lm[1].y*h),     # Nose
        (lm[33].x*w, lm[33].y*h),   # Left eye
        (lm[263].x*w, lm[263].y*h), # Right eye
        (lm[61].x*w, lm[61].y*h),   # Left mouth
        (lm[291].x*w, lm[291].y*h)  # Right mouth
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (-30.0, -30.0, -30.0),
        (30.0, -30.0, -30.0),
        (-40.0, 30.0, -30.0),
        (40.0, 30.0, -30.0)
    ])

    focal_length = w
    center = (w/2, h/2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1))

    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return False

    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)

    yaw = angles[1]

    return abs(yaw) < 15
 # threshold

# ---- Process folder ----

frontal_images = []

files = [f for f in os.listdir(INPUT_FOLDER)
         if f.lower().endswith((".jpg",".png",".jpeg"))]

for file in tqdm(files):
    path = os.path.join(INPUT_FOLDER, file)
    img = cv2.imread(path)

    if img is None:
        continue

    if is_frontal(img):
        frontal_images.append(file)

# ---- Save CSV ----

df = pd.DataFrame({"image_id": frontal_images})
df.to_csv(CSV_OUTPUT, index=False)

print(f"Saved {len(frontal_images)} frontal images to CSV.")
