import cv2
import os

# ---------------------------------
# 1️⃣ Input & Output Folders
# ---------------------------------
input_folder = "nose_only"          # folder with grayscale nose crops
output_folder = "nose_sharpened"    # folder to save sharpened images

os.makedirs(output_folder, exist_ok=True)

# ---------------------------------
# 2️⃣ CLAHE Object (created once)
# ---------------------------------
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

# ---------------------------------
# 3️⃣ Process All Images
# ---------------------------------
for filename in os.listdir(input_folder):

    img_path = os.path.join(input_folder, filename)

    # Read as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Skipped:", filename)
        continue

    # ---------------------------------
    # Optional: Resize for consistency
    # ---------------------------------
    img = cv2.resize(img, (64, 64))

    # ---------------------------------
    # 1️⃣ Improve Contrast using CLAHE
    # ---------------------------------
    enhanced = clahe.apply(img)

    # ---------------------------------
    # 2️⃣ Apply Unsharp Mask
    # ---------------------------------
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(enhanced, 1.6, blur, -0.6, 0)

    # ---------------------------------
    # 3️⃣ Save sharpened image
    # ---------------------------------
    save_path = os.path.join(output_folder, filename)
    cv2.imwrite(save_path, sharpened)

print("✅ All nose images sharpened successfully.")
