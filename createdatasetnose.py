import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# --- CONFIG ---
INPUT_DIR = r'D:\major_phase2\nose_subset_images'
BASE_OUTPUT = r'D:\major_phase2\nose_dataset'
folders = ['sketches', 'masks', 'outlines']
for f in folders: os.makedirs(os.path.join(BASE_OUTPUT, f), exist_ok=True)

CROP_SIZE = 256
PADDING = 10   # Reduced to prevent mask from reaching eyes/mouth
DARKNESS = 3.0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# --- PRECISE NOSE INDICES ---
# These landmarks focus strictly on the bridge, tip, and nostrils.
NOSE_INDICES = [
    1, 2, 3, 4, 5, 6, 168, 197, 195, 5, 4, 98, 97, 326, 327, 
    102, 129, 209, 49, 331, 358, 429, 279, 196, 419, 458, 238
]



def to_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (45, 45), 0)
    sketch = cv2.divide(gray, 255 - blur + 1, scale=256.0)
    return (np.power(sketch/255.0, DARKNESS) * 255).astype(np.uint8)

def process():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not files:
        print(f"No images found in {INPUT_DIR}")
        return

    for filename in tqdm(files):
        img = cv2.imread(os.path.join(INPUT_DIR, filename))
        if img is None: continue
        h, w, _ = img.shape
        
        # Convert to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb_img)
        
        if not res.multi_face_landmarks: continue

        # 1. Create a tight Mask for the Nose only
        mask = np.zeros((h, w), dtype=np.uint8)
        landmarks = res.multi_face_landmarks[0].landmark
        
        # Extract only the nose points
        pts = []
        for i in NOSE_INDICES:
            pt = landmarks[i]
            pts.append((int(pt.x * w), int(pt.y * h)))
        
        pts = np.array(pts)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # Slight dilation to capture nostril edges without reaching eyes
        mask = cv2.dilate(mask, np.ones((PADDING, PADDING)), iterations=1)

        # 2. Precise Crop Logic
        # Center the crop on the nose tip (landmark index 1)
        tip = landmarks[1]
        cx, cy = int(tip.x * w), int(tip.y * h)
        
        x1, y1 = cx - CROP_SIZE // 2, cy - CROP_SIZE // 2
        x2, y2 = x1 + CROP_SIZE, y1 + CROP_SIZE

        # Ensure we stay within image bounds (padding if necessary)
        # Using white (255) for background padding in sketches
        canvas = np.ones((h + CROP_SIZE, w + CROP_SIZE, 3), dtype=np.uint8) * 255
        canvas[CROP_SIZE//2:h+CROP_SIZE//2, CROP_SIZE//2:w+CROP_SIZE//2] = img
        
        mask_canvas = np.zeros((h + CROP_SIZE, w + CROP_SIZE), dtype=np.uint8)
        mask_canvas[CROP_SIZE//2:h+CROP_SIZE//2, CROP_SIZE//2:w+CROP_SIZE//2] = mask
        
        # Recalculate crop coordinates for the padded canvas
        cx_p, cy_p = cx + CROP_SIZE//2, cy + CROP_SIZE//2
        crop_img = canvas[cy_p-CROP_SIZE//2 : cy_p+CROP_SIZE//2, 
                          cx_p-CROP_SIZE//2 : cx_p+CROP_SIZE//2]
        crop_mask = mask_canvas[cy_p-CROP_SIZE//2 : cy_p+CROP_SIZE//2, 
                                cx_p-CROP_SIZE//2 : cx_p+CROP_SIZE//2]

        # 3. Apply Forensic Transformation
        # Black out everything outside the mask before sketching
        isolated_nose = np.where(cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR) == 255, crop_img, 255)
        
        sketch = to_sketch(isolated_nose)
        sketch[crop_mask == 0] = 255  # Solid white background for forensic clarity
        
        # Canny edge detection for the Outline (the "skeleton" for the GAN)
        # We use lower thresholds because nose lines are softer than eye lines
        outline = cv2.Canny(sketch, 40, 120) 
        outline = 255 - outline # Invert: Black lines on White

        # 4. Final Save
        cv2.imwrite(os.path.join(BASE_OUTPUT, 'sketches', filename), sketch)
        cv2.imwrite(os.path.join(BASE_OUTPUT, 'masks', filename), crop_mask)
        cv2.imwrite(os.path.join(BASE_OUTPUT, 'outlines', filename), outline)

if __name__ == "__main__":
    process()
    print("\n--- Nose Dataset Generation Complete ---")