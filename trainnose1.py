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
class NoseForensicDataset(Dataset):
    def __init__(self, csv_file, sketch_dir, outline_dir, mask_dir, img_size=128):
        self.df = pd.read_csv(csv_file)
        self.sketch_dir = sketch_dir
        self.outline_dir = outline_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] for Tanh
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']
        
        # Load Images as Grayscale ('L')
        sketch = self.transform(Image.open(os.path.join(self.sketch_dir, img_id)).convert('L'))
        outline = self.transform(Image.open(os.path.join(self.outline_dir, img_id)).convert('L'))
        mask = self.transform(Image.open(os.path.join(self.mask_dir, img_id)).convert('L'))
        
        # Load Attributes (Male, Young, etc.)
        attrs = torch.tensor(self.df.iloc[idx, 1:].values.astype('float32'))
        return sketch, outline, mask, attrs

# ==========================================
# 2. HIGH-DETAIL ARCHITECTURE (ResNet + PatchGAN)
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
        # Encoder
        self.enc1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(2, 64, 7, 1), nn.InstanceNorm2d(64), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True))
        
        # Style Bridge
        self.style_bridge = nn.Sequential(
            nn.Linear(attr_dim + latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 256 * 32 * 32),
            nn.ReLU(True)
        )
        
        # 6 ResBlocks for detailing
        self.bottleneck = nn.Sequential(*[ResBlock(512) for _ in range(6)])
        
        # Decoder
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

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 4, 1, 1), nn.Sigmoid()
        )
    def forward(self, o, m, s):
        return self.model(torch.cat([o, m, s], 1))

# ==========================================
# 3. INPUT & EXECUTION SECTION
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    CSV_PATH = r"D:\major_phase2\nose_final_train.csv"
    SKETCH_DIR = r"D:\major_phase2\nose_sketch_images"
    OUTLINE_DIR = r"D:\major_phase2\perfect_outlines_nose"
    MASK_DIR = r"D:\major_phase2\exact_nose_masks"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8
    EPOCHS = 50
    
    # --- Initialize Data ---
    dataset = NoseForensicDataset(CSV_PATH, SKETCH_DIR, OUTLINE_DIR, MASK_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Initialize Models ---
    netG = HighDetailGenerator().to(DEVICE)
    netD = PatchDiscriminator().to(DEVICE)
    
    optG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()

    

    # --- Training Loop ---
    print(f"Training started on {DEVICE}...")
    for epoch in range(EPOCHS):
        for i, (sketches, outlines, masks, attrs) in enumerate(loader):
            sketches, outlines, masks, attrs = sketches.to(DEVICE), outlines.to(DEVICE), masks.to(DEVICE), attrs.to(DEVICE)
            latent = torch.randn(sketches.size(0), 64).to(DEVICE)

            # Train Discriminator
            optD.zero_grad()
            real_patch = netD(outlines, masks, sketches)
            loss_D_real = criterion_GAN(real_patch, torch.ones_like(real_patch))
            
            fake_sketches = netG(outlines, masks, attrs, latent)
            fake_patch = netD(outlines, masks, fake_sketches.detach())
            loss_D_fake = criterion_GAN(fake_patch, torch.zeros_like(fake_patch))
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optD.step()

            # Train Generator
            optG.zero_grad()
            g_patch = netD(outlines, masks, fake_sketches)
            loss_G_GAN = criterion_GAN(g_patch, torch.ones_like(g_patch))
            loss_G_L1 = criterion_L1(fake_sketches, sketches) * 100 # Detail weight
            
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optG.step()

        print(f"Epoch [{epoch}/{EPOCHS}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    # --- Save Model ---
    torch.save(netG.state_dict(), "high_detail_nose_model.pth")
    print("Model saved successfully.")