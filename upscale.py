import cv2
import os
import pandas as pd

CSV = "frontal_images.csv"
INPUT =r"D:\major_phase2\masks"
OUTPUT = "resizedmask"

SIZE = 256  # or 512

os.makedirs(OUTPUT, exist_ok=True)

df = pd.read_csv(CSV)

for name in df["image_id"]:
    path = os.path.join(INPUT,name.replace(".jpg",".png"))

    img = cv2.imread(path)

    if img is None:
        continue

    resized = cv2.resize(
        img,
        (SIZE, SIZE),
        interpolation=cv2.INTER_LANCZOS4
    )

    cv2.imwrite(os.path.join(OUTPUT, name), resized)

print("Done")
