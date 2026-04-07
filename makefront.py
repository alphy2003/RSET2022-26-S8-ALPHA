import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = r"D:\major_phase2\filtered_eye_images1"
YAW_LIMIT = 12   
PITCH_LIMIT = 12 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True
)

def get_head_pose(image):
    img_h, img_w, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None, None 

    face_landmarks = results.multi_face_landmarks[0].landmark
    face_3d = []
    face_2d = []

    # Using 6 key points for stable pose estimation
    indices = [1, 152, 33, 263, 61, 291]

    for idx in indices:
        lm = face_landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])

    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    rmat, _ = cv2.Rodrigues(rot_vec)
    
    # FIX: Catch all 7 values from the function
    proj_matrix = np.hstack((rmat, trans_vec))
    # mtxC, mtxR, tvec, mtxRX, mtxRY, mtxRZ, eulerAngles
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch = angles[0] * 360
    yaw = angles[1] * 360
    
    return yaw, pitch

def clean_non_frontal_images():
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(('.jpg', '.png'))]
    removed_count = 0
    not_detected = 0
    
    print(f"Analyzing {len(files)} images...")

    for f in tqdm(files):
        img_path = os.path.join(DATA_DIR, f)
        image = cv2.imread(img_path)
        
        if image is None: continue

        try:
            yaw, pitch = get_head_pose(image)

            if yaw is None:
                os.remove(img_path)
                not_detected += 1
                continue

            # Check limits
            if abs(yaw) > YAW_LIMIT or abs(pitch) > PITCH_LIMIT:
                os.remove(img_path)
                removed_count += 1
        except Exception as e:
            # If an image is corrupt or causes a math error, remove it to be safe
            os.remove(img_path)
            removed_count += 1

    print(f"\n--- Cleanup Report ---")
    print(f"Deleted (Too much turn/tilt): {removed_count}")
    print(f"Deleted (Face not found): {not_detected}")
    print(f"Remaining: {len(os.listdir(DATA_DIR))}")

if __name__ == "__main__":
    clean_non_frontal_images()