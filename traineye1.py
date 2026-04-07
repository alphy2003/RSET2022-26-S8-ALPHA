import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

# ==========================================
# 1. DATA LOADING SECTION
# ==========================================
class ForensicDataset(Dataset):
    def __init__(self, csv_file, sketch_dir, outline_dir, mask_dir, img_size=128):
        self.df = pd.read_csv(csv_file)
        self.sketch_dir = sketch_dir
        self.outline_dir = outline_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']
        sketch = self.transform(Image.open(os.path.join(self.sketch_dir, img_id)).convert('L'))
        outline = self.transform(Image.open(os.path.join(self.outline_dir, img_id)).convert('L'))
        mask = self.transform(Image.open(os.path.join(self.mask_dir, img_id)).convert('L'))
        # Convert attributes to float tensor
        attrs = torch.tensor(self.df.iloc[idx, 1:].values.astype('float32'))
        return sketch, outline, mask, attrs

# ==========================================
# 2. GENERATOR SECTION (U-Net + Style Bridge)
# ==========================================
class ForensicGenerator(nn.Module):
    def __init__(self, attr_dim, latent_dim=64):
        super(ForensicGenerator, self).__init__()
        # Encoder: Input 2 channels (Outline + Mask)
        self.enc1 = nn.Conv2d(2, 64, 4, 2, 1) 
        self.enc2 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128))
        self.enc3 = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256))
        
        # Style Bridge: Merges Latent Vector + Attributes
        self.style_bridge = nn.Sequential(
            nn.Linear(attr_dim + latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 16 * 16), 
            nn.ReLU()
        )
        # Decoder
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.final = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh())

    def forward(self, outline, mask, attrs, latent):
        s1 = self.enc1(torch.cat([outline, mask], dim=1))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        style = self.style_bridge(torch.cat([attrs, latent], dim=1)).view(-1, 256, 16, 16)
        d1 = self.dec1(torch.cat([s3, style], dim=1))
        d2 = self.dec2(torch.cat([d1, s2], dim=1))
        return self.final(torch.cat([d2, s1], dim=1))

# ==========================================
# 3. DISCRIMINATOR SECTION
# ==========================================
class ForensicDiscriminator(nn.Module):
    def __init__(self):
        super(ForensicDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), # Outline + Mask + Sketch
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 1), nn.Sigmoid()
        )
    def forward(self, outline, mask, sketch):
        return self.model(torch.cat([outline, mask, sketch], dim=1))

# ==========================================
# 4. TRAINING EXECUTION SECTION
# ==========================================
# Config
ATTR_DIM = 6 # Number of attribute columns in your CSV
LATENT_DIM = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Models
netG = ForensicGenerator(ATTR_DIM, LATENT_DIM).to(device)
netD = ForensicDiscriminator().to(device)

# Optimizers & Loss
optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

# Load Data
dataset = ForensicDataset(
    csv_file=r"D:\major_phase2\train.csv",
    sketch_dir=r"D:\major_phase2\intense_sketch_images",
    outline_dir=r"D:\major_phase2\perfect_outlines",
    mask_dir=r"D:\major_phase2\exact_eye_masks"
)
loader = DataLoader(dataset, batch_size=16, shuffle=True)



print("Starting Training...")
for epoch in range(50):
    for sketches, outlines, masks, attrs in loader:
        sketches, outlines, masks, attrs = sketches.to(device), outlines.to(device), masks.to(device), attrs.to(device)
        latent = torch.randn(outlines.size(0), LATENT_DIM).to(device)

        # --- Train Discriminator ---
        optD.zero_grad()
        real_label = torch.ones_like(netD(outlines, masks, sketches))
        fake_label = torch.zeros_like(real_label)
        
        loss_D_real = criterion_GAN(netD(outlines, masks, sketches), real_label)
        fake_sketches = netG(outlines, masks, attrs, latent)
        loss_D_fake = criterion_GAN(netD(outlines, masks, fake_sketches.detach()), fake_label)
        
        (loss_D_real + loss_D_fake).backward()
        optD.step()

        # --- Train Generator ---
        optG.zero_grad()
        loss_G_GAN = criterion_GAN(netD(outlines, masks, fake_sketches), real_label)
        loss_G_L1 = criterion_L1(fake_sketches, sketches) * 100 # High weight for exactness
        
        (loss_G_GAN + loss_G_L1).backward()
        optG.step()

    print(f"Epoch [{epoch}/100] Loss D: {loss_D_real+loss_D_fake:.4f} Loss G: {loss_G_GAN+loss_G_L1:.4f}")

torch.save(netG.state_dict(), "forensic_generator.pth")