import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image

# --- CONFIGURATION ---
CSV_FILE = "frontal_images.csv"      # Path to your CSV file
ID_COLUMN = "image_id"            # The name of the column containing the IDs
INPUT_DIR = r"D:\Dataset\img_align_celeba\img_align_celeba"      # Folder where the original images are stored
OUTPUT_DIR = "sketches_output"    # Folder to save the sketches
IMAGE_EXTENSION = ".jpg"          # Or .png, depending on your files
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. Load the CSV
df = pd.read_csv(CSV_FILE)

# 2. Load the Pipeline (Optimized for speed: SDXL-Turbo + Canny)
print("Loading AI models (Offline)...")
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", 
    torch_dtype=torch.float16
).to(DEVICE)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16"
).to(DEVICE)

# Memory optimizations for faster inference
pipe.enable_model_cpu_offload() 

def generate_sketch(image_path, save_path):
    # Load and prep image (1024x1024 is standard for SDXL)
    image = load_image(image_path).resize((1024, 1024))
    
    # Create Canny Edge (The Structural Skeleton)
    image_np = np.array(image)
    image_canny = cv2.Canny(image_np, 100, 200)
    image_canny = image_canny[:, :, None]
    image_canny = np.concatenate([image_canny, image_canny, image_canny], axis=2)
    canny_image = Image.fromarray(image_canny)

    # The Prompt for the style you requested
    prompt = "high-quality professional charcoal sketch, clean graphite pencil lines, minimalist, white background, hatching, artistic masterpiece, sharp edges, hand-drawn"
    negative_prompt = "color, photo, realistic, 3d render, blurry, messy, low quality"

    # 1-Step generation for maximum speed
    output = pipe(
        prompt,
        controlnet_conditioning_scale=0.8,
        image=canny_image,
        num_inference_steps=1, 
        guidance_scale=0.0,
        negative_prompt=negative_prompt
    ).images[0]

    output.save(save_path)

# --- BATCH PROCESSING FROM CSV ---
print(f"Processing {len(df)} images from {CSV_FILE}...")

for index, row in df.iterrows():
    img_id = str(row[ID_COLUMN])
    
    # Construct full filename (adds extension if missing)
    filename = img_id if img_id.endswith(IMAGE_EXTENSION) else f"{img_id}{IMAGE_EXTENSION}"
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, f"sketch_{filename}")

    if os.path.exists(input_path):
        print(f"[{index+1}/{len(df)}] Processing: {filename}")
        try:
            generate_sketch(input_path, output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"Skipping: {filename} (File not found in {INPUT_DIR})")

print(f"\nSuccessfully finished! Your sketches are in '{OUTPUT_DIR}'.")