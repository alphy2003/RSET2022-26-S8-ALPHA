import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
MASTER_IMG_DIR = r"D:\Dataset\img_align_celeba\img_align_celeba"
SAVE_MASK_DIR = r"D:\major_phase2\nose_masks_full"
CSV_PATH = r"D:\major_phase2\nose_dataset_cleaned.csv"

if not os.path.exists(SAVE_MASK_DIR):
    os.makedirs(SAVE_MASK_DIR)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Points that define the nose area
NOSE_INDICES = [168, 6, 197, 195, 5, 4, 1, 2, 98, 327, 326, 97, 129, 217, 437, 358]

def run_masking():
    df = pd.read_csv(CSV_PATH)
    valid_list = []

    print(f"Starting mask generation for {len(df)} images...")

    for img_name in tqdm(df['image_id']):
        img_path = os.path.join(MASTER_IMG_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None: continue
        
        h, w, _ = image.shape
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            mask = np.zeros((h, w), dtype=np.uint8)
            points = []
            for face_landmarks in results.multi_face_landmarks:
                for idx in NOSE_INDICES:
                    lm = face_landmarks.landmark[idx]
                    points.append([int(lm.x * w), int(lm.y * h)])
            
            # Create a smooth shape for the nose mask
            hull = cv2.convexHull(np.array(points, dtype=np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Save the mask
            cv2.imwrite(os.path.join(SAVE_MASK_DIR, img_name), mask)
            valid_list.append(img_name)

    # Save a final CSV that only contains images where the mask was successfully created
    df_final = df[df['image_id'].isin(valid_list)]
    df_final.to_csv(r"D:\major_phase2\nose_final_train.csv", index=False)
    print(f"\nDone! Created {len(valid_list)} masks.")
    print(f"Final training list saved to: nose_final_train.csv")

if __name__ == "__main__":
    run_masking()