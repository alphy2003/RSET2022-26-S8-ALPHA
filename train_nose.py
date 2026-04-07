import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import os
import numpy as np
import time

# --- CONFIGURATION ---
CSV_PATH = r"D:\major_phase2\nose_final_train.csv"
IMG_DIR = r"D:\major_phase2\final_anatomic_composites"
CHECKPOINT_DIR = r"D:\major_phase2\checkpoints_final_nose"
LOG_FILE = os.path.join(CHECKPOINT_DIR, "training_losses.csv")
LATENT_DIM = 100
ATTR_DIM = 5  # [Big_Nose, Pointy_Nose, Male, Young, Pale_Skin]
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100
SAVE_INTERVAL = 5 

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- 1. ARCHITECTURE ---

class ResBlock(nn.Module):
    """Residual Block to preserve high-detail line work"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x): 
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: Noise + Attributes
        self.init = nn.Sequential(
            nn.Linear(LATENT_DIM + ATTR_DIM, 256 * 32 * 32),
            nn.ReLU(True)
        )
        self.conv_blocks = nn.Sequential(
            nn.Unflatten(1, (256, 32, 32)),
            ResBlock(256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResBlock(128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 128x128
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh() # Output range [-1, 1]
        )

    def forward(self, z, labels):
        # Concatenate latent noise and attribute vector
        x = torch.cat([z, labels], 1)
        x = self.init(x)
        return self.conv_blocks(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1), # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1),
            nn.Sigmoid()
        )
    def forward(self, img): 
        return self.model(img)

# --- 2. DATA UTILITIES ---

class NoseDataset(Dataset):
    def __init__(self, csv_path, img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.attrs = self.df[['Big_Nose', 'Pointy_Nose', 'Male', 'Young', 'Pale_Skin']].values.astype(np.float32)
        self.filenames = self.df['image_id'].values

    def __len__(self): 
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load merged sketch as grayscale
        img_name = f"final_{self.filenames[idx]}"
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path, 0)
        
        if img is None: # Fallback if image missing
            img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        # Normalize to [-1, 1] for Tanh activation
        img = torch.FloatTensor(img).unsqueeze(0) / 127.5 - 1.0
        return img, torch.FloatTensor(self.attrs[idx])

def get_distribution_weights(csv_path):
    df = pd.read_csv(csv_path)
    weights = []
    for col in ['Big_Nose', 'Pointy_Nose', 'Male', 'Young', 'Pale_Skin']:
        pos_count = df[col].sum()
        total = len(df)
        # Higher weight for rarer classes like Pale_Skin
        weight = total / (2.0 * pos_count) if pos_count > 0 else 1.0
        weights.append(weight)
    return torch.FloatTensor(weights)

# --- 3. TRAINING INITIALIZATION ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
disc = Discriminator().to(device)

# Optimization
g_optimizer = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss Functions
adversarial_loss = nn.BCELoss()
l1_loss = nn.L1Loss() # The "High-Detail" anchor

# Prepare Logger
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("epoch,d_loss,g_adv_loss,g_pixel_loss,total_g_loss\n")

dataset = NoseDataset(CSV_PATH, IMG_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 4. MAIN TRAINING LOOP ---

print(f"Training started on {device}...")

for epoch in range(EPOCHS):
    epoch_d_loss = 0
    epoch_g_adv = 0
    epoch_g_pixel = 0
    start_time = time.time()

    gen.train()
    for i, (real_imgs, labels) in enumerate(loader):
        batch_size = real_imgs.size(0)
        real_imgs, labels = real_imgs.to(device), labels.to(device)
        
        # Ground Truth labels
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        d_optimizer.zero_grad()
        
        # Generate fake images based on real labels (Conditional)
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_imgs = gen(z, labels)
        
        real_loss = adversarial_loss(disc(real_imgs), valid)
        fake_loss = adversarial_loss(disc(fake_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        d_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        g_optimizer.zero_grad()
        
        # 1. Adversarial Loss: Can it fool the Discriminator?
        g_adv = adversarial_loss(disc(fake_imgs), valid)
        
        # 2. Pixel Loss: Does it match the actual sketch details?
        # High-Detail Multiplier (100x) ensures lines are sharp
        g_pixel = l1_loss(fake_imgs, real_imgs)
        
        total_g_loss = g_adv + (100 * g_pixel)
        
        total_g_loss.backward()
        g_optimizer.step()

        # Stats
        epoch_d_loss += d_loss.item()
        epoch_g_adv += g_adv.item()
        epoch_g_pixel += g_pixel.item()

    # Epoch Summary
    avg_d = epoch_d_loss / len(loader)
    avg_g_adv = epoch_g_adv / len(loader)
    avg_g_pix = epoch_g_pixel / len(loader)
    avg_total_g = avg_g_adv + (100 * avg_g_pix)
    
    print(f"Epoch [{epoch}/{EPOCHS}] | D Loss: {avg_d:.4f} | G Adv: {avg_g_adv:.4f} | G Pixel: {avg_g_pix:.4f} | Time: {time.time()-start_time:.1f}s")

    # Log Losses
    with open(LOG_FILE, "a") as f:
        f.write(f"{epoch},{avg_d:.4f},{avg_g_adv:.4f},{avg_g_pix:.4f},{avg_total_g:.4f}\n")

    # --- 5. CHECKPOINTING ---
    if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS - 1:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'g_opt': g_optimizer.state_dict(),
            'd_opt': d_optimizer.state_dict(),
        }, checkpoint_path)
        print(f"--> Saved checkpoint: {checkpoint_path}")

print("Training Complete.")