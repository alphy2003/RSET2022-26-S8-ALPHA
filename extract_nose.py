import pandas as pd
import cv2
import os

# ---------------------------------
# 1️⃣ Load filtered CSV
# ---------------------------------
df_images = pd.read_csv("filtered_no_eyeglasses.csv")

# ---------------------------------
# 2️⃣ Load landmarks CSV
# ---------------------------------
landmarks = pd.read_csv(r"D:\Dataset\list_landmarks_align_celeba.csv")

# Clean column names (remove spaces if any)
df_images.columns = df_images.columns.str.strip()
landmarks.columns = landmarks.columns.str.strip()

# ---------------------------------
# 3️⃣ Merge on image_id
# ---------------------------------
df = pd.merge(df_images, landmarks, on="image_id", how="inner")

print("Total images to process:", len(df))

# ---------------------------------
# 4️⃣ Paths
# ---------------------------------
image_folder = "sketches2" # folder containing original images
output_folder = "nose_only"

os.makedirs(output_folder, exist_ok=True)

# ---------------------------------
# 5️⃣ Extract nose region
# ---------------------------------
for _, row in df.iterrows():

    img_path = os.path.join(image_folder, row["image_id"])
    img = cv2.imread(img_path)

    if img is None:
        print("Image not found:", img_path)
        continue

    nose_x = int(row["nose_x"])
    nose_y = int(row["nose_y"])

    left_eye_x = int(row["lefteye_x"])
    right_eye_x = int(row["righteye_x"])

    h, w, _ = img.shape

    # Crop size proportional to eye distance
    eye_distance = abs(left_eye_x - right_eye_x)
    crop_size = int(eye_distance * 0.35)

    x1 = max(0, nose_x - crop_size)
    x2 = min(w, nose_x + crop_size)
    y1 = max(0, nose_y - crop_size)
    y2 = min(h, nose_y + crop_size)

    nose_crop = img[y1:y2, x1:x2]

    save_path = os.path.join(output_folder, row["image_id"])
    cv2.imwrite(save_path, nose_crop)

print("✅ Nose extraction completed successfully.")
