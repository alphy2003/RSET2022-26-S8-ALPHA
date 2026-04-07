import os
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

############################################
# PATHS (EDIT THESE)
############################################

CSV_PATH = r"D:\major_phase2\hair_attributes1.csv"
IMAGE_FOLDER = r"D:\Dataset\img_align_celeba\img_align_celeba"
OUTPUT_FOLDER = r"D:\major_phase2\hair_only"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

############################################
# LOAD MODEL
############################################

print("Loading SegFormer model...")

processor = SegformerImageProcessor.from_pretrained(
    "jonathandinu/face-parsing"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "jonathandinu/face-parsing"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"Running on: {device}")

############################################
# READ CSV
############################################

df = pd.read_csv(CSV_PATH)

# assumes column name is 'image_id'
image_list = df["image_id"].tolist()

############################################
# HAIR CLASS ID
############################################

HAIR_ID = 13  # hair label in this model

############################################
# PROCESS LOOP
############################################

for img_name in tqdm(image_list):

    img_path = os.path.join(IMAGE_FOLDER, img_name)

    if not os.path.exists(img_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ##################################
    # Segmentation
    ##################################

    inputs = processor(images=rgb, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    seg = outputs.logits.argmax(dim=1)[0].cpu().numpy()

    ##################################
    # Hair mask
    ##################################

    hair_mask = (seg == HAIR_ID).astype(np.uint8) * 255

    ##################################
    # Resize mask to original image
    ##################################

    hair_mask = cv2.resize(
        hair_mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    ##################################
    # Extract hair only
    ##################################

    hair_only = cv2.bitwise_and(image, image, mask=hair_mask)

    ##################################
    # Save result
    ##################################

    save_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(save_path, hair_only)

print("\n✅ DONE — Hair extracted for all images!")
