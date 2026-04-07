import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

INPUT_DIR = r"D:\major_phase2\noseimagetest"
OUTLINE_DIR = r"D:\major_phase2\perfect_outlines_nose_test"
os.makedirs(OUTLINE_DIR, exist_ok=True)

# Indices for the outer silhouette of the face in Mediapipe
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def get_perfect_outline(image_path):
    image = cv2.imread(image_path)
    if image is None: return
    h, w, _ = image.shape
    
    # Convert to RGB for Mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    # Create a pure white background
    mask = np.ones((h, w), dtype=np.uint8) * 255
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Collect coordinates of the oval landmarks
            points = []
            for idx in FACE_OVAL:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append((x, y))
            
            # Draw the clean face outline
            points = np.array(points, np.int32)
            cv2.polylines(mask, [points], isClosed=True, color=(0), thickness=2)
            
    return mask

# Process all images
for filename in os.listdir(INPUT_DIR):
    outline = get_perfect_outline(os.path.join(INPUT_DIR, filename))
    if outline is not None:
        cv2.imwrite(os.path.join(OUTLINE_DIR, filename), outline)

print("Perfect forensic outlines generated successfully.")