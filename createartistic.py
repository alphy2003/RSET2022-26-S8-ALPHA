import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import LineartDetector

# -----------------------------
# PATHS (EDIT THESE)
# -----------------------------
csv_path = "frontal_images.csv"
input_folder = "D:\Dataset\img_align_celeba\img_align_celeba"
output_folder = "output_sketches"

os.makedirs(output_folder, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD CSV
# -----------------------------
df = pd.read_csv(csv_path)

# Convert image_id column to string
image_ids = df["image_id"].astype(str).tolist()

print(f"Found {len(image_ids)} image IDs in CSV")

# -----------------------------
# LOAD MODELS ONCE
# -----------------------------
print("Loading models...")

processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_lineart",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

pipe.safety_checker = None

print("Models loaded successfully!")

# -----------------------------
# PROMPTS
# -----------------------------
prompt = """
clean pencil sketch, bold contour lines,
strong black outlines, high contrast line art,
white paper background, accurate face,
minimal shading
"""

negative_prompt = """
color, painting, watercolor, oil painting,
engraving, heavy etching, dark background,
abstract patterns, distorted face,
extra lines, dramatic shadows
"""

# -----------------------------
# PROCESS IMAGES BASED ON CSV
# -----------------------------
for img_id in image_ids:

    # Try common extensions
    possible_files = [
        f"{img_id}.jpg",
        f"{img_id}.png",
        f"{img_id}.jpeg"
    ]

    input_path = None

    for img_name in image_ids:

        img_name = img_name.strip()  # remove extra spaces

        input_path = os.path.join(input_folder, img_name)

        if not os.path.exists(input_path):
            print(f"Image not found for ID: {img_name}")
            continue

        print(f"Processing: {img_name}")

        # Load image
        input_image = Image.open(input_path).convert("RGB")

        # Generate lineart
        control_image = processor(input_image)

        # Strengthen edges
        control_np = np.array(control_image)
        control_np = cv2.convertScaleAbs(control_np, alpha=1.8, beta=0)
        control_image = Image.fromarray(control_np)

        # Generate sketch
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=30,
            guidance_scale=6.5,
            controlnet_conditioning_scale=1.2
        ).images[0]

        # Save output
        output_path = os.path.join(output_folder, img_name)
        image.save(output_path)
