import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# --- 1. ARCHITECTURE (Must match your training script exactly) ---
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, attr_dim=5):
        super().__init__()
        self.init = nn.Sequential(
            nn.Linear(latent_dim + attr_dim, 256 * 32 * 32),
            nn.ReLU(True)
        )
        self.conv_blocks = nn.Sequential(
            nn.Unflatten(1, (256, 32, 32)),
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, labels], 1)
        x = self.init(x)
        return self.conv_blocks(x)

# --- 2. INTENTION HANDLER ---
class ForensicSystem:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = Generator().to(self.device)
        
        # Load the weights
        print(f"Loading weights from {model_path}...")
        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt['gen_state_dict'] if 'gen_state_dict' in ckpt else ckpt
        self.gen.load_state_dict(state_dict)
        self.gen.eval()

    def process_intention(self, text):
        """Translates user text into the 5-digit binary code your model learned"""
        text = text.lower()
        # [Big_Nose, Pointy_Nose, Male, Young, Pale_Skin]
        vector = [0, 0, 0, 0, 0]
        
        if "big" in text or "wide" in text: vector[0] = 1
        if "pointy" in text or "sharp" in text: vector[1] = 1
        if "male" in text or "man" in text: vector[2] = 1
        if "young" in text or "child" in text: vector[3] = 1
        if "pale" in text or "fair" in text: vector[4] = 1
        
        return torch.FloatTensor([vector]).to(self.device)

    def run_test(self, user_intention):
        attr_vector = self.process_intention(user_intention)
        results = []

        print(f"Generating variations for intention: {user_intention}...")
        
        # Generate 8 variations with 8 different seeds
        for s in range(8):
            torch.manual_seed(s * 100) # Changing the seed manually
            z = torch.randn(1, 100).to(self.device)
            
            with torch.no_grad():
                fake_img = self.gen(z, attr_vector)
            
            # Convert to viewable image
            img = ((fake_img.squeeze().cpu().numpy() + 1) * 127.5).astype(np.uint8)
            results.append(img)

        # Stitch them into a 2x4 grid
        top_row = np.hstack(results[:4])
        bottom_row = np.hstack(results[4:])
        final_grid = np.vstack([top_row, bottom_row])
        
        cv2.imwrite("intention_result.png", final_grid)
        print("Success! Result saved as 'intention_result.png'")

# --- 3. RUN ---
if __name__ == "__main__":
    MODEL_PATH = r"D:\major_phase2\checkpoints_final_nose\checkpoint_epoch_35.pth"
    
    if os.path.exists(MODEL_PATH):
        system = ForensicSystem(MODEL_PATH)
        
        # CHANGE YOUR INTENTION HERE:
        user_input = "a big pale female nose"
        
        system.run_test(user_input)
    else:
        print("Checkpoint file not found!")