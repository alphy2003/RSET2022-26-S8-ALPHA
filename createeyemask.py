import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# ==== SETTINGS ====
INPUT_FOLDER = r"D:\Dataset\img_align_celeba\img_align_celeba"
OUTPUT_MASK_FOLDER = "masks"
OUTPUT_EYE_FOLDER = "eyes"   # optional
SAVE_EYE_REGION = True       # set False if only masks needed

os.makedirs(OUTPUT_MASK_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_EYE_FOLDER, exist_ok=True)

# Eye landmark indices
LEFT_EYE = [33,133,160,159,158,157,173,246]
RIGHT_EYE = [362,263,387,386,385,384,398,466]

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

image_files = [f for f in os.listdir(INPUT_FOLDER)
               if f.lower().endswith(('.png','.jpg','.jpeg'))]

for file in tqdm(image_files):
    path = os.path.join(INPUT_FOLDER, file)
    img = cv2.imread(path)

    if img is None:
        continue

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    mask = np.zeros((h, w), dtype=np.uint8)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]

        for eye in [LEFT_EYE, RIGHT_EYE]:
            pts = []
            for idx in eye:
                lm = landmarks.landmark[idx]
                x, y = int(lm.x*w), int(lm.y*h)
                pts.append([x,y])

            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask, [pts], 255)

    # Save mask
    mask_path = os.path.join(
        OUTPUT_MASK_FOLDER,
        os.path.splitext(file)[0] + ".png"
    )
    cv2.imwrite(mask_path, mask)

    # Optionally save extracted eye region
    if SAVE_EYE_REGION:
        eye_region = cv2.bitwise_and(img, img, mask=mask)
        eye_path = os.path.join(
            OUTPUT_EYE_FOLDER,
            os.path.splitext(file)[0] + ".png"
        )
        cv2.imwrite(eye_path, eye_region)

print("Done!")
