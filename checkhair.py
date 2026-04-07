import cv2
import pandas as pd
import os
import random

CSV = "hair_length_labels.csv"
IMAGE_FOLDER = r"D:\Dataset\img_align_celeba\img_align_celeba"

df = pd.read_csv(CSV)

for _ in range(100):

    row = df.sample(1).iloc[0]

    path = os.path.join(IMAGE_FOLDER,row.image_id)

    img = cv2.imread(path)
    if img is None:
        continue

    text = f"Length: {row.hair_length}"

    cv2.putText(img,text,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Check",img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
