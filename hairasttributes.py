import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

SKETCH_FOLDER = r"D:\major_phase2\hair_sketch"
OUTPUT_CSV = r"D:\major_phase2\hair_attributes2.csv"

results = []

for file in tqdm(os.listdir(SKETCH_FOLDER)):
    if not file.lower().endswith((".jpg",".png",".jpeg")):
        continue
    
    path = os.path.join(SKETCH_FOLDER,file)
    
    img = cv2.imread(path,0)  # grayscale
    h,w = img.shape
    
    # Binary threshold
    _,mask = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
    
    hair_pixels = np.sum(mask>0)
    total_pixels = h*w
    
    # -------- AREA --------
    area_ratio = hair_pixels/total_pixels
    
    if area_ratio < 0.08:
        area_label="thin"
    elif area_ratio < 0.18:
        area_label="normal"
    else:
        area_label="thick"
    
    # -------- LENGTH --------
    ys = np.where(mask>0)[0]
    
    if len(ys)==0:
        continue
    
    top = ys.min()
    bottom = ys.max()
    
    length_ratio = (bottom-top)/h
    
    if length_ratio < 0.35:
        length_label="short"
    elif length_ratio < 0.6:
        length_label="medium"
    else:
        length_label="long"
    
    results.append([file,length_label,area_label])

df = pd.DataFrame(results,columns=["image","length","volume"])
df.to_csv(OUTPUT_CSV,index=False)

print("✅ Done!")
