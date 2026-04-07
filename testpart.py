import torch
import torch.nn as nn
from torchvision import utils
import os

# --- 1. CONFIGURATION (Must match Training Script) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTR_DIM = 6        # Arched, Bushy, Narrow, Bags, Spects, Male
LATENT_DIM = 100    
IMG_SIZE = 128      
CHECKPOINT_PATH = "forensic_eye_checkpoint.pth"

# --- 2. GENERATOR ARCHITECTURE (Must match Training Script) ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = IMG_SIZE // 16 
        self.l1 = nn.Sequential(nn.Linear(LATENT_DIM + ATTR_DIM, 512 * self.init_size**2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, attrs):
        combined = torch.cat([z, attrs], dim=1)
        out = self.l1(combined)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        return self.conv_blocks(out)

# --- 3. TESTING / INFERENCE FUNCTION ---
def generate_suspect_eye(prompt):
    """
    Translates text into a forensic image.
    Example prompt: "male bushy bags"
    """
    # Create the keyword mapping (ensure this matches your CSV column order!)
    mapping = ["arched", "bushy", "narrow", "bags", "spects", "male"]
    attr_vector = [0.0] * ATTR_DIM
    
    # Simple NLP: check if the keyword is in the user's prompt
    words = prompt.lower().split()
    for word in words:
        if word in mapping:
            idx = mapping.index(word)
            attr_vector[idx] = 1.0
            
    attr_tensor = torch.tensor([attr_vector], dtype=torch.float).to(DEVICE)

    # Load Model
    if not os.path.exists(CHECKPOINT_PATH):
        print("Error: Checkpoint file not found. Train the model first!")
        return

    netG = Generator().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    netG.load_state_dict(checkpoint['G_state'])
    netG.eval()

    # Generate Image from Random Noise + Attributes
    # You can change the 'seed' to get different versions of the same description
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    
    with torch.no_grad():
        generated_img = netG(z, attr_tensor)
    
    # Save the result
    output_filename = f"generated_{prompt.replace(' ', '_')}.png"
    utils.save_image(generated_img, output_filename, normalize=True)
    print(f"--- SUCCESS ---")
    print(f"Prompt: {prompt}")
    print(f"Vector used: {attr_vector}")
    print(f"Result saved as: {output_filename}")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    user_input = input("Enter eye description (e.g., 'male bushy bags'): ")
    generate_suspect_eye(user_input)