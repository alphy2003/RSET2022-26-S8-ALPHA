import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

#########################################
# LOAD MODEL
#########################################

print("Loading model...")

processor = SegformerImageProcessor.from_pretrained(
    "jonathandinu/face-parsing"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "jonathandinu/face-parsing"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("✅ Model Loaded")

#########################################
# PATHS
#########################################

IMAGE_DIR = r"D:\Dataset\img_align_celeba\img_align_celeba"
CSV_PATH = r"D:\major_phase2\front_facing_hair_dataset.csv"

df = pd.read_csv(CSV_PATH)

#########################################
# HAIR MASK FUNCTION
#########################################

def get_hair_mask(img):

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled = torch.nn.functional.interpolate(
        logits,
        size=rgb.shape[:2],
        mode="bilinear",
        align_corners=False
    )

    parsing = upsampled.argmax(dim=1)[0].cpu().numpy()

    # hair class = 17
    mask = (parsing == 17).astype(np.uint8) * 255

    return mask

#########################################
# ATTRIBUTE FUNCTIONS
#########################################

def hair_length(mask):
    ys = np.where(mask > 0)[0]

    if len(ys) == 0:
        return "unknown"

    length = ys.max() - ys.min()

    if length < 60:
        return "short"
    elif length < 140:
        return "medium"
    else:
        return "long"


def hair_volume(mask):
    area = np.sum(mask > 0)

    if area < 5000:
        return "thin"
    elif area < 15000:
        return "normal"
    else:
        return "thick"


def hair_part(mask):
    h, w = mask.shape

    top = mask[0:int(h*0.25), :]

    left = np.sum(top[:, :w//3])
    center = np.sum(top[:, w//3:2*w//3])
    right = np.sum(top[:, 2*w//3:])

    if center < left*0.5 and center < right*0.5:
        return "middle"
    elif left < right:
        return "left"
    else:
        return "right"

#########################################
# PROCESS DATASET
#########################################

results = []

print("Processing images...")

for img_name in tqdm(df["image_id"]):

    path = os.path.join(IMAGE_DIR, img_name)

    img = cv2.imread(path)

    if img is None:
        continue

    mask = get_hair_mask(img)

    length = hair_length(mask)
    volume = hair_volume(mask)
    part = hair_part(mask)

    results.append([img_name, length, volume, part])

#########################################
# SAVE CSV
#########################################

out_df = pd.DataFrame(
    results,
    columns=["image_id", "hair_length", "hair_volume", "hair_part"]
)

out_df.to_csv("hair_attributes1.csv", index=False)

print("✅ Done! Saved hair_attributes.csv")
