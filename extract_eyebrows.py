import os
import cv2
import numpy as np
import mediapipe as mp

INPUT_FOLDER = r"D:\major_phase2\sketches_512"
OUTPUT_FOLDER = "eyebrow_crops"
OUTPUT_SIZE = (128, 64)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52]
RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295, 282]

for img_name in os.listdir(INPUT_FOLDER):

    img_path = os.path.join(INPUT_FOLDER, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        continue

    h, w, _ = image.shape
    landmarks = results.multi_face_landmarks[0]

    brow_points = []

    for idx in LEFT_BROW + RIGHT_BROW:
        lm = landmarks.landmark[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        brow_points.append([x, y])

    brow_points = np.array(brow_points)

    x_min = np.min(brow_points[:, 0])
    x_max = np.max(brow_points[:, 0])
    y_min = np.min(brow_points[:, 1])
    y_max = np.max(brow_points[:, 1])

    # Adaptive margins
    brow_height = y_max - y_min
    brow_width = x_max - x_min

    x_margin = int(0.25 * brow_width)
    y_margin_top = int(1.5 * brow_height)
    y_margin_bottom = int(0.8 * brow_height)

    x1 = max(0, x_min - x_margin)
    x2 = min(w, x_max + x_margin)
    y1 = max(0, y_min - y_margin_top)
    y2 = min(h, y_max + y_margin_bottom)

    crop = image[y1:y2, x1:x2]

    crop = cv2.resize(crop, OUTPUT_SIZE)

    save_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(save_path, crop)

print("High-quality eyebrow extraction completed.")