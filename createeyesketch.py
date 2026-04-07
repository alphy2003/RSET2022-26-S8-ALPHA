import cv2
import os
import glob
import numpy as np

def create_intense_sketch(img_path, save_path):
    # Load image
    img = cv2.imread(img_path)
    if img is None: 
        print(f"Error: Could not load {img_path}")
        return
    
    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Invert Gray Image
    inverted_gray = 255 - gray
    
    # Step 3: Blur the inverted image
    # Note: Reduced kernel from 21 to 13 for sharper lines
    blurred = cv2.GaussianBlur(inverted_gray, (13, 13), 0)
    
    # Step 4: Blend Gray and Blurred (Basic Sketch)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    
    # Step 5: Increase Intensity using CLAHE
    # This enhances local contrast to make nostrils and edges "pop"
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    sketch = clahe.apply(sketch)
    
    # Step 6: Darken the lines (Gamma Correction)
    # A gamma < 1.0 darkens the pencil strokes significantly
    gamma = 0.7
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    intense_sketch = cv2.LUT(sketch, table)
    
    # Optional: Convert back to 3 channels (RGB) for GAN compatibility
    sketch_3ch = cv2.cvtColor(intense_sketch, cv2.COLOR_GRAY2RGB)
    
    cv2.imwrite(save_path, sketch_3ch)

# --- EXECUTION ---
input_folder = r"D:\major_phase2\eye_images_12k"
output_folder = r"D:\major_phase2\intense_eye_images_12k"
os.makedirs(output_folder, exist_ok=True)

print("Processing images...")
image_files = glob.glob(os.path.join(input_folder, "*.jpg"))

for img_file in image_files:
    name = os.path.basename(img_file)
    create_intense_sketch(img_file, os.path.join(output_folder, name))

print(f"Intensity upgrade complete! {len(image_files)} sketches saved to: {output_folder}")