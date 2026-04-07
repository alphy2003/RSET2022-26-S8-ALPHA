import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# ----------- PATHS -----------
ALIGNED_DIR = "aligned_images_128"
MASK_OUT_DIR = "aligned_eye_masks"
FAILED_LIST = "failed_mask_generation.txt"

os.makedirs(MASK_OUT_DIR, exist_ok=True)

# -------- Mediapipe ---------
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# -------- Eye + Eyebrow Landmark IDs --------
LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]

LEFT_EYEBROW = [70,63,105,66,107,55,65,52]
RIGHT_EYEBROW = [336,296,334,293,300,276,283,282]

EYE_REGION = LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW


def generate_eye_mask(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = FACE_MESH.process(rgb)

    if not result.multi_face_landmarks:
        return None

    pts = []
    for lm_id in EYE_REGION:
        lm = result.multi_face_landmarks[0].landmark[lm_id]
        x, y = int(lm.x * w), int(lm.y * h)
        pts.append((x, y))

    if len(pts) < 10:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(np.array(pts))
    cv2.fillConvexPoly(mask, hull, 255)

    return mask


# ------------ MAIN LOOP --------------
failed = []

print("\n🎨 Generating TRUE eye masks for aligned images...\n")

for fn in tqdm(os.listdir(ALIGNED_DIR)):
    if not fn.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(ALIGNED_DIR, fn)
    mask = generate_eye_mask(img_path)

    if mask is None:
        failed.append(fn)
        continue

    out_path = os.path.join(MASK_OUT_DIR, fn.replace(".jpg", ".png"))
    cv2.imwrite(out_path, mask)

# ------- SAVE FAILURES -------
with open(FAILED_LIST, "w") as f:
    for fn in failed:
        f.write(fn + "\n")

print("\n✅ Done!")
print(f"🟢 Successful masks: {len(os.listdir(MASK_OUT_DIR))}")
print(f"🔴 Failed masks: {len(failed)}")
