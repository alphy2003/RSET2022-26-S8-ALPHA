import torch
import cv2
import numpy as np
from diffusers import StableDiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
).to(DEVICE)

pipe.safety_checker = None  # optional

def generate_hair(prompt):
    full_prompt = f"""
    black and white pencil sketch of hairstyle,
    clean lines, minimal shading,
    white background,
    {prompt}
    """

    image = pipe(
        full_prompt,
        guidance_scale=8.5,
        num_inference_steps=30
    ).images[0]

    img = np.array(image)

    # Convert to sketch style
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv,(21,21),0)
    sketch = cv2.divide(gray,255-blur,scale=256)

    cv2.imwrite("hair_result.png", sketch)
    print("✅ Saved as hair_result.png")


while True:
    text = input("\nDescribe hair: ")
    generate_hair(text)
