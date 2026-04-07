import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
import os

# ======================
# CONFIG
# ======================
IMAGE_SIZE = 128
NOISE_DIM = 100
ATTR_DIM = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Point this to your best performing epoch (e.g., epoch_50.pth)
CHECKPOINT_PATH = "checkpoints_eyeonly/epoch_16.pth" 
OUTPUT_FOLDER = "forensic_outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ======================
# UPDATED GENERATOR (Matches your new architecture)
# ======================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(NOISE_DIM + ATTR_DIM, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 4 * 4 * 1024), 
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 3, 3, 1, 1), 
            nn.Tanh()
        )

    def forward(self, noise, attrs):
        x = torch.cat([noise, attrs], dim=1)
        x = self.fc(x)
        x = x.view(-1, 1024, 4, 4)
        return self.net(x)

# ======================
# INFERENCE LOGIC
# ======================
def test_generation(attribute_vector, filename="result_eye.png"):
    # 1. Load Model
    model = Generator().to(DEVICE)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['G'])
        print(f"Loaded weights from {CHECKPOINT_PATH}")
    except FileNotFoundError:
        print("Error: Checkpoint file not found. Ensure training has saved at least one epoch.")
        return

    model.eval()

    # 2. Setup Inputs
    # attributes must be float and have shape (1, 12)
    test_attr = torch.tensor([attribute_vector]).float().to(DEVICE)
    noise = torch.randn(1, NOISE_DIM).to(DEVICE)

    # 3. Generate
    with torch.no_grad():
        generated_img = model(noise, test_attr)

    # 4. Denormalize (-1 to 1  -->  0 to 1)
    generated_img = (generated_img + 1.0) / 2.0
    
    # 5. Save and Upscale
    temp_path = os.path.join(OUTPUT_FOLDER, "temp_128.png")
    final_path = os.path.join(OUTPUT_FOLDER, filename)
    
    vutils.save_image(generated_img, temp_path)
    
    # Using Lanczos or Bicubic for high-quality forensic upscaling
    with Image.open(temp_path) as img:
        img_512 = img.resize((512, 512), Image.Resampling.LANCZOS)
        img_512.save(final_path)
    
    os.remove(temp_path) # Clean up the 128px version
    print(f"Forensic eye saved to: {final_path}")

# ======================
# RUN TEST
# ======================
if __name__ == "__main__":
    # Example attributes for an eye (binary: 0 or 1)
    # Define based on your CSV columns (e.g. Brown, Blue, Hazel, Narrow, etc.)
    forensic_description = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0] 
    
    test_generation(forensic_description, "high_res_eye_v1.png")