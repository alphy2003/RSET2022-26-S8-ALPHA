import os
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import LineartDetector

# -------- SETTINGS --------
input_folder = r"D:\Dataset\img_align_celeba\img_align_celeba"
output_folder = "sketches1"
os.makedirs(output_folder, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

prompt = """
pen and ink portrait,
cross-hatching drawing,
forensic artist sketch,
black ink lines,
white paper,
highly detailed
"""
# --------------------------

print("Loading ControlNet...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_lineart",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)



pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
).to(device)

pipe.enable_attention_slicing()

print("Loading lineart detector...")
detector = LineartDetector.from_pretrained("lllyasviel/Annotators")

print("Processing images...")

for file in os.listdir(input_folder):

    if not file.lower().endswith((".jpg",".jpeg",".png")):
        continue

    img_path = os.path.join(input_folder, file)
    image = Image.open(img_path).convert("RGB").resize((512,512))

    # Extract line art
    lineart = detector(image)

    # Generate sketch
    result = pipe(
        prompt=prompt,
        image=lineart,
        guidance_scale=9,
        num_inference_steps=30
    ).images[0]

    result.save(os.path.join(output_folder, file))

    print("Done:", file)

print("\n✅ All sketches generated!")
