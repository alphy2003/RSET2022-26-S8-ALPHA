import torch
import torch.nn as nn
from torchvision import utils, transforms
from PIL import Image
import os

# ==========================================
# 1. GENERATOR ARCHITECTURE (Same as Training)
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x): return x + self.block(x)

class HighDetailGenerator(nn.Module):
    def __init__(self, attr_dim=5, latent_dim=64):
        super(HighDetailGenerator, self).__init__()
        self.enc1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(2, 64, 7, 1), nn.InstanceNorm2d(64), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True))
        self.style_bridge = nn.Sequential(
            nn.Linear(attr_dim + latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256 * 32 * 32),
            nn.ReLU(True)
        )
        self.bottleneck = nn.Sequential(*[ResBlock(512) for _ in range(6)])
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), nn.InstanceNorm2d(256), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.InstanceNorm2d(128), nn.ReLU(True))
        self.final = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(128, 1, 7), nn.Tanh())

    def forward(self, outline, mask, attrs, latent):
        s1 = self.enc1(torch.cat([outline, mask], 1))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        style = self.style_bridge(torch.cat([attrs, latent], 1)).view(-1, 256, 32, 32)
        mid = self.bottleneck(torch.cat([s3, style], 1))
        return self.final(self.dec2(self.dec1(mid)))

# ==========================================
# 2. USER PROMPT TRANSLATOR
# ==========================================
def translate_prompt_to_vector(user_text):
    """
    Translates user text into a 5-digit attribute vector.
    Assumes CSV Order: Male, Young, Pointy, Wide, Flat
    """
    # Mapping keywords to indices
    mapping = {
        "male": 0, "man": 0, "boy": 0,
        "young": 1, "child": 1, "youth": 1,
        "pointy": 2, "sharp": 2, "thin": 2,
        "wide": 3, "broad": 3, "large": 3,
        "flat": 4, "low": 4, "blunt": 4
    }
    
    attr_vector = [0.0] * 5
    words = user_text.lower().replace(",", " ").split()
    
    for word in words:
        if word in mapping:
            attr_vector[mapping[word]] = 1.0
            
    return attr_vector

# ==========================================
# 3. MAIN INFERENCE SCRIPT
# ==========================================
def run_forensic_test():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "high_detail_nose_model.pth"
    
    # User Inputs
    print("\n--- Forensic Nose Generator ---")
    user_prompt = input("Enter description (e.g., 'male wide flat nose'): ")
    outline_img_path = input("Enter path to face structure outline: ")
    mask_img_path = input("Enter path to face structure mask: ")

    # 1. Translate Text
    attr_list = translate_prompt_to_vector(user_prompt)
    attr_tensor = torch.tensor([attr_list], dtype=torch.float32).to(DEVICE)
    
    # 2. Preprocess Images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    outline = transform(Image.open(outline_img_path).convert('L')).unsqueeze(0).to(DEVICE)
    mask = transform(Image.open(mask_img_path).convert('L')).unsqueeze(0).to(DEVICE)
    latent = torch.randn(1, 64).to(DEVICE) # Variations

    # 3. Load Model and Generate
    netG = HighDetailGenerator(attr_dim=5).to(DEVICE)
    netG.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    netG.eval()

    with torch.no_grad():
        output = netG(outline, mask, attr_tensor, latent)
    
    # 4. Save
    save_name = f"result_{user_prompt.replace(' ', '_')}.png"
    utils.save_image(output, save_name, normalize=True)
    print(f"\n[SUCCESS] Vector Generated: {attr_list}")
    print(f"[SUCCESS] Image saved as: {save_name}")

if __name__ == "__main__":
    run_forensic_test()