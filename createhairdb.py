import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# Initialize MediaPipe
mp_selfie = mp.solutions.selfie_segmentation
mp_face_mesh = mp.solutions.face_mesh

def generate_hair_to_neck_mask(image_dir, csv_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = pd.read_csv(csv_path)
    image_ids = df['image_id'].tolist()

    with mp_selfie.SelfieSegmentation(model_selection=0) as selfie, \
         mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:

        for img_id in tqdm(image_ids):
            path = os.path.join(image_dir, img_id)
            img = cv2.imread(path)
            if img is None: continue
            h, w, _ = img.shape

            # 1. Get the full person silhouette
            res_s = selfie.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            person_mask = (res_s.segmentation_mask > 0.5).astype(np.uint8) * 255

            # 2. Identify the Face Area
            face_mask = np.zeros((h, w), dtype=np.uint8)
            res_m = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if res_m.multi_face_landmarks:
                landmarks = res_m.multi_face_landmarks[0].landmark
                
                # Face Oval indices
                face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                
                points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in face_oval])
                cv2.fillPoly(face_mask, [points], 255)

                # 3. Define the Neck Cutoff
                # We take the chin (152) and add a percentage of the face height to reach the neck
                chin_y = int(landmarks[152].y * h)
                forehead_y = int(landmarks[10].y * h)
                face_height = chin_y - forehead_y
                
                # Neck base is roughly 25% of face height below the chin
                neck_cutoff_y = chin_y + int(face_height * 0.25)

                # 4. Apply the Cutoff to the Person Silhouette
                # This removes shoulders/torso below the neck line
                if neck_cutoff_y < h:
                    person_mask[neck_cutoff_y:, :] = 0

            # 5. Final Result: Person (up to neck) MINUS Face
            hair_neck_mask = cv2.subtract(person_mask, face_mask)

            # 6. Save
            mask_path = os.path.join(save_dir, img_id.replace('.', '_hair_neck.'))
            cv2.imwrite(mask_path, hair_neck_mask)

generate_hair_to_neck_mask(r"D:\Dataset\img_align_celeba\img_align_celeba", r"D:\major_phase2\front_facing_hair_dataset.csv", 'hair_masks')