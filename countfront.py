import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def count_front_facing(csv_path, img_dir, output_csv, yaw_threshold=15, pitch_threshold=15):
    # Load the dataset
    df = pd.read_csv(csv_path)
    image_ids = df['image_id'].tolist()
    
    front_facing_images = []
    total_processed = 0
    front_count = 0

    print(f"Starting analysis of {len(image_ids)} images...")

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for img_id in tqdm(image_ids):
            img_path = os.path.join(img_dir, img_id)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
            
            h, w, _ = image.shape
            # Convert to RGB for MediaPipe
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # 3D model points for pose estimation
                face_2d = []
                face_3d = []
                
                # Critical landmarks: Nose tip, Chin, Left Eye corner, Right Eye corner, Mouth corners
                indices = [33, 263, 1, 61, 291, 199] 
                for idx in indices:
                    lm = landmarks[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # Camera Matrix approximation (Simplified)
                focal_length = 1 * w
                cam_matrix = np.array([[focal_length, 0, w / 2],
                                      [0, focal_length, h / 2],
                                      [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP to find rotation
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                
                # Convert rotation vector to rotation matrix
                rmat, _ = cv2.Rodrigues(rot_vec)
                
                # Decompose projection matrix
                # FIX: Added 7 underscores/variables to handle the OpenCV return mismatch
                res = cv2.decomposeProjectionMatrix(np.hstack((rmat, trans_vec)))
                
                # Flatten the angles to ensure we have a simple 1D array [pitch, yaw, roll]
                angles = res[0].flatten() # The first element contains Euler angles

                # Use .item() or [0][0] to get the scalar value from the array
                pitch = angles[0].item()
                yaw = angles[1].item()
                roll = angles[2].item()

                # Now the comparison will work perfectly
                if abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold:
                    front_count += 1
                    front_facing_images.append(img_id)
            
            total_processed += 1

    # Save results to a new filtered CSV
    filtered_df = df[df['image_id'].isin(front_facing_images)]
    filtered_df.to_csv(output_csv, index=False)

    print(f"\n--- Analysis Complete ---")
    print(f"Total Processed: {total_processed}")
    print(f"Front-Facing Count: {front_count}")
    print(f"Filtered Dataset Saved To: {output_csv}")

if __name__ == "__main__":
    # SET YOUR PATHS HERE
    IMAGE_DIRECTORY = r"D:\Dataset\img_align_celeba\img_align_celeba"
    INPUT_CSV = r"D:\major_phase2\clean_hair_dataset.csv"
    OUTPUT_CSV = r"D:\major_phase2\front_facing_hair_dataset.csv"

    count_front_facing(INPUT_CSV, IMAGE_DIRECTORY, OUTPUT_CSV)