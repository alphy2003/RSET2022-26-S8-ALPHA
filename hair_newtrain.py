import torch
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading ControlNet...")

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
).to(DEVICE)

pipe.safety_checker = None

canny = CannyDetector()

print("✅ Ready!")

def generate_hair(prompt):
    
    # Create a blank guide image
    blank = np.ones((512,512,3), dtype=np.uint8)*255
    
    # Edge map from blank (acts like sketch canvas)
    edges = canny(blank, 100, 200)
    edges = Image.fromarray(edges)

    full_prompt = f"""
    black and white pencil sketch of hairstyle,
    clean lines,
    white background,
    {prompt}
    """

    result = pipe(
        prompt=full_prompt,
        image=edges,
        guidance_scale=9,
        num_inference_steps=30
    ).images[0]

    result.save("hair_result.png")
    print("✅ Saved as hair_result.png")


while True:
    text = input("\nDescribe hair: ")
    generate_hair(text)
