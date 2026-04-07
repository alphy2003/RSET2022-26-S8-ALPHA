import cv2
import mediapipe as mp
import numpy as np
import os
from math import atan2, degrees

mp_face = mp.solutions.face_mesh

INPUT_DIR = r"selected_images"       # your selected 111k files
OUTPUT_DIR = r"aligned_images_128"         # output aligned folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

def align_face(image, landmarks):
    # Get left and right eye keypoints (mediapipe indices)
    LEFT_EYE = [33, 133]   # center approx
    RIGHT_EYE = [362, 263]

    h, w, _ = image.shape

    left = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right = np.array([landmarks[263].x * w, landmarks[263].y * h])

    # Angle between eyes
    delta_y = right[1] - left[1]
    delta_x = right[0] - left[0]
    angle = atan2(delta_y, delta_x)
    angle_deg = degrees(angle)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(tuple(left), angle_deg, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def crop_center(image, size=178):  
    h, w, _ = image.shape

    cx, cy = w // 2, h // 2
    half = size // 2

    return image[cy-half:cy+half, cx-half:cx+half]

def process_image(path, output_path):
    img = cv2.imread(path)
    if img is None:
        return False

    with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not res.multi_face_landmarks:
            return False

        lm = res.multi_face_landmarks[0].landmark
        aligned = align_face(img, lm)

        try:
            cropped = crop_center(aligned)
            resized = cv2.resize(cropped, (128, 128))
        except:
            return False

        cv2.imwrite(output_path, resized)
        return True


# ----------------------

print("🚀 Aligning images...")

count = 0
failed = []

for fname in os.listdir(INPUT_DIR):
    in_path = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, fname)

    success = process_image(in_path, out_path)
    if not success:
        failed.append(fname)
    else:
        count += 1

print(f"\n✅ Alignment complete: {count} images saved.")
print(f"❌ Failed alignments: {len(failed)}")
print("Failed list saved to failed_alignments.txt")

with open("failed_alignments.txt", "w") as f:
    for name in failed:
        f.write(name + "\n")
