import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# Initialize MediaPipe
mp_selfie = mp.solutions.selfie_segmentation
mp_face_mesh = mp.solutions.face_mesh

def generate_hair_masks(image_dir, csv_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = pd.read_csv(csv_path)
    image_ids = df['image_id'].tolist()

    # selfie_segmentation finds the general person
    # face_mesh finds the exact face to subtract it
    with mp_selfie.SelfieSegmentation(model_selection=0) as selfie, \
         mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:

        print("Generating Hair-to-Neck Masks...")
        for img_id in tqdm(image_ids):
            img_path = os.path.join(image_dir, img_id)
            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape

            # 1. Get the full person silhouette (includes hair and neck)
            res_s = selfie.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            person_mask = (res_s.segmentation_mask > 0.5).astype(np.uint8) * 255

            # 2. Identify the Face Area to subtract it
            face_mask = np.zeros((h, w), dtype=np.uint8)
            res_m = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if res_m.multi_face_landmarks:
                landmarks = res_m.multi_face_landmarks[0].landmark
                
                # Face Oval indices (Standard MediaPipe face boundary)
                face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                
                points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in face_oval])
                cv2.fillPoly(face_mask, [points], 255)

                # 3. Create Neck Cutoff
                # We find the chin (152) and forehead (10) to calculate face height
                chin_y = int(landmarks[152].y * h)
                forehead_y = int(landmarks[10].y * h)
                neck_limit = chin_y + int((chin_y - forehead_y) * 0.25) # 25% below chin

                # Remove everything below the neck line (shoulders/torso)
                if neck_limit < h:
                    person_mask[neck_limit:, :] = 0

            # 4. Final Hair Mask = (Person up to neck) MINUS (Face)
            final_mask = cv2.subtract(person_mask, face_mask)

            # 5. Save (Use .png for masks to avoid quality loss)
            mask_name = img_id.split('.')[0] + '.png'
            cv2.imwrite(os.path.join(save_dir, mask_name), final_mask)

if __name__ == "__main__":
    ORIGINAL_IMG_DIR = r"D:\Dataset\img_align_celeba\img_align_celeba"
    CSV_PATH = r"D:\major_phase2\front_facing_hair_dataset.csv"
    MASK_SAVE_DIR = r"D:\major_phase2\hair_masks"

    generate_hair_masks(ORIGINAL_IMG_DIR, CSV_PATH, MASK_SAVE_DIR)