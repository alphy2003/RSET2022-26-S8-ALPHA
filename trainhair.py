import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

########################################
# SETTINGS
########################################
CSV_PATH = r"D:\major_phase2\balanced_hair_dataset.csv"
IMG_DIR = r"D:\major_phase2\hair_sketch"
IMG_SIZE = 128
BATCH_SIZE = 16 # Reduced batch size often helps stability with InstanceNorm
EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ATTR_COLS = [
    "Bald", "length_short", "length_medium", "length_long",
    "volume_thin", "volume_normal", "volume_thick",
    "Straight_Hair", "Wavy_Hair"
]

########################################
# DATASET (Optimized)
########################################
class HairDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Pass attributes as a vector (1D) instead of constant maps
        attr_vector = torch.tensor(row[ATTR_COLS].values.astype(np.float32))

        # Sketch image handling
        img_path = os.path.join(self.img_dir, row["image_id"])
        img = cv2.imread(img_path, 0)
        if img is None: # Fallback for missing files
            img = np.ones((IMG_SIZE, IMG_SIZE)) * 255
            
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = (img / 127.5) - 1.0 # Scale to [-1, 1]
        img = torch.tensor(img).unsqueeze(0).float()

        return attr_vector, img

########################################
# GENERATOR (U-Net with Attribute Injection)
########################################
def conv_block(in_c, out_c, down=True):
    if down:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c), # Changed out_f to out_c
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_c), # Changed out_f to out_c
            nn.ReLU(inplace=True)
        )

class Generator(nn.Module):
    def __init__(self, attr_dim=9):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2)) # Input is Noise/Seed
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        # Attribute MLP (to inject into bottleneck)
        self.attr_mlp = nn.Sequential(
            nn.Linear(attr_dim, 512),
            nn.ReLU()
        )

        # Decoder
        self.dec1 = conv_block(1024, 256, down=False) # 512 (enc4) + 512 (attr)
        self.dec2 = conv_block(512, 128, down=False)  # 256 (dec1) + 256 (enc3)
        self.dec3 = conv_block(256, 64, down=False)   # 128 (dec2) + 128 (enc2)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, attr):
        # Initial spatial seed (noise)
        e1 = self.enc1(noise)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Inject attributes into bottleneck
        a = self.attr_mlp(attr).view(-1, 512, 1, 1)
        a = a.expand(-1, -1, e4.size(2), e4.size(3))
        
        # Concatenate encoded image features with attributes
        bottleneck = torch.cat([e4, a], 1)

        d1 = self.dec1(bottleneck)
        d2 = self.dec2(torch.cat([d1, e3], 1))
        d3 = self.dec3(torch.cat([d2, e2], 1))
        
        return self.final(torch.cat([d3, e1], 1))

########################################
# DISCRIMINATOR
########################################
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, img):
        return self.model(img)

########################################
# TRAINING ENGINE
########################################
dataset = HairDataset(CSV_PATH, IMG_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

# Slowed down D to allow G to learn the attributes
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.999))

bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()

os.makedirs("output_samples", exist_ok=True)

for epoch in range(EPOCHS):
    G.train()
    D.train()
    
    for i, (attr, real) in enumerate(loader):
        attr, real = attr.to(DEVICE), real.to(DEVICE)
        
        # Create a spatial noise seed (gives the model "texture" to start with)
        noise_seed = torch.randn(real.size(0), 1, IMG_SIZE, IMG_SIZE).to(DEVICE)

        # Train Discriminator
        fake = G(noise_seed, attr)
        d_real = D(real)
        d_fake = D(fake.detach())
        
        lossD = (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))) / 2
        
        optD.zero_grad()
        lossD.backward()
        optD.step()

        # Train Generator
        d_fake_g = D(fake)
        lossG = bce(d_fake_g, torch.ones_like(d_fake_g)) + 150 * l1(fake, real) # Weighted L1 for lines
        
        optG.zero_grad()
        lossG.backward()
        optG.step()

    # Visual Inspection (Crucial to see if diversity is returning)
    if epoch % 5 == 0:
        save_image(fake[0] * 0.5 + 0.5, f"output_samples/epoch_{epoch}.png")
        print(f"Epoch {epoch} | LossG: {lossG.item():.4f} | LossD: {lossD.item():.4f}")

    # Checkpoint
    torch.save(G.state_dict(), f"generator_v2_epoch_{epoch}.pth")