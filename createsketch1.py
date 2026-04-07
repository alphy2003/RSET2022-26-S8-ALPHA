import cv2
import os
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FOLDER = r"D:\Dataset\img_align_celeba\img_align_celeba"
OUTPUT_FOLDER = 'sketches2'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created directory: {OUTPUT_FOLDER}")

def to_artistic_sketch(img, blur_ksize=45, darkness_factor=5.0):
    """
    Converts image to sketch with adjustable darkness.
    
    Parameters:
    - blur_ksize: (Odd Number) Controls line thickness. Higher = Thicker.
    - darkness_factor: (Float) Controls line darkness. 
                       1.0 is default. >1.0 makes lines darker.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Invert the grayscale image
    inverted_gray = 255 - gray
    
    # 3. Blur the inverted image
    # We use a large kernel (45) to make the strokes thick enough to be visible
    blurred = cv2.GaussianBlur(inverted_gray, (blur_ksize, blur_ksize), 0)
    
    # 4. Invert the blurred image back
    inverted_blurred = 255 - blurred
    
    # 5. Create the initial sketch (Color Dodge)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    
    # --- STEP 6: DARKEN THE EDGES (Gamma Correction) ---
    # We normalize to 0-1 range first
    sketch_float = sketch / 255.0
    
    # Apply power law (Gamma > 1 makes mid-tones darker)
    # This turns light gray lines into dark black lines
    sketch_darkened = np.power(sketch_float, darkness_factor)
    
    # Scale back to 0-255 and convert to integer
    sketch_final = (sketch_darkened * 255).astype(np.uint8)
    
    return sketch_final

# --- MAIN PROCESSING LOOP ---
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

# Check if input folder exists
if not os.path.exists(INPUT_FOLDER):
    print(f"Error: Input folder not found at {INPUT_FOLDER}")
else:
    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]
    print(f"Found {len(all_files)} images to process.")

    for filename in tqdm(all_files, desc="Converting to Dark Sketches"):
        try:
            img_path = os.path.join(INPUT_FOLDER, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Process with increased darkness
            sketch = to_artistic_sketch(img)
            
            save_path = os.path.join(OUTPUT_FOLDER, filename)
            cv2.imwrite(save_path, sketch)
            
        except Exception as e:
            print(f"Error on {filename}: {e}")

    print("\nDone! Sketches saved in:", OUTPUT_FOLDER)
