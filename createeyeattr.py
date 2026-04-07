

import cv2
import mediapipe as mp
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# =========================
# CONFIG
# =========================
IMAGE_FOLDER = r"D:\Dataset\img_align_celeba\img_align_celeba"
OUTPUT_CSV = "eye_derived_attributes.csv"

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe eye landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

data = []

with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True) as face_mesh:

    for filename in tqdm(os.listdir(IMAGE_FOLDER)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(IMAGE_FOLDER, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image.shape

        def get_eye_points(indices):
            pts = []
            for idx in indices:
                lm = landmarks.landmark[idx]
                pts.append([lm.x * w, lm.y * h])
            return np.array(pts)

        left_eye = get_eye_points(LEFT_EYE_IDX)
        right_eye = get_eye_points(RIGHT_EYE_IDX)

        # -------- Eye Shape (EAR) --------
        def compute_ear(eye):
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            return (A + B) / (2.0 * C)

        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2

        # -------- Orientation (Slope) --------
        left_slope = left_eye[3][1] - left_eye[0][1]
        right_slope = right_eye[3][1] - right_eye[0][1]
        avg_slope = (left_slope + right_slope) / 2

        # -------- Hooded (Vertical Distance) --------
        def eyelid_distance(eye):
            upper_mid = (eye[1] + eye[2]) / 2
            lower_mid = (eye[4] + eye[5]) / 2
            return np.linalg.norm(upper_mid - lower_mid)

        left_vdist = eyelid_distance(left_eye)
        right_vdist = eyelid_distance(right_eye)
        avg_vdist = (left_vdist + right_vdist) / 2

        data.append({
            "image_id": filename,
            "ear": avg_ear,
            "slope": avg_slope,
            "vertical_dist": avg_vdist
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# =========================
# PERCENTILE-BASED LABELING
# =========================

# Eye Shape thresholds
ear_low = np.percentile(df["ear"], 33)
ear_high = np.percentile(df["ear"], 66)

def classify_shape(ear):
    if ear < ear_low:
        return "Monolid"
    elif ear < ear_high:
        return "Almond"
    else:
        return "Round"

df["eye_shape"] = df["ear"].apply(classify_shape)

# Orientation thresholds
slope_low = np.percentile(df["slope"], 33)
slope_high = np.percentile(df["slope"], 66)

def classify_orientation(s):
    if s < slope_low:
        return "Upturned"
    elif s > slope_high:
        return "Downturned"
    else:
        return "Neutral"

df["orientation"] = df["slope"].apply(classify_orientation)

# Hooded threshold
hood_thresh = np.percentile(df["vertical_dist"], 50)

def classify_hooded(v):
    return "Yes" if v < hood_thresh else "No"

df["hooded"] = df["vertical_dist"].apply(classify_hooded)

# Save final attributes
final_df = df[["image_id", "eye_shape", "orientation", "hooded"]]
final_df.to_csv(OUTPUT_CSV, index=False)

print("Final Eye Attribute CSV Created!")
print("\nDistribution Check:")
print(final_df["eye_shape"].value_counts(normalize=True) * 100)
print(final_df["orientation"].value_counts(normalize=True) * 100)
print(final_df["hooded"].value_counts(normalize=True) * 100)