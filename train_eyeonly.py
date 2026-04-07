import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils

# ======================
# CONFIG
# ======================
IMAGE_SIZE = 256
NOISE_DIM = 128
ATTR_DIM = 12
BATCH_SIZE = 16
EPOCHS = 100

LR_G = 0.0001
LR_D = 0.0003

R1_GAMMA = 10
EMA_BETA = 0.999

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

os.makedirs("samples_256", exist_ok=True)
os.makedirs("checkpoints_256", exist_ok=True)

# ======================
# DATASET
# ======================
class EyeDataset(Dataset):
    def __init__(self, csv_file, image_folder):
        self.df = pd.read_csv(csv_file)
        self.image_folder = image_folder

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["image_id"]).split('.')[0]
        img_path = os.path.join(self.image_folder, f"{image_id}_merged.jpg")

        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_folder, f"{image_id}.jpg")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        attrs = torch.tensor(row[1:].values.astype(np.float32))
        return image, attrs

# ======================
# GENERATOR (256)
# ======================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(NOISE_DIM + ATTR_DIM, 4 * 4 * 1024),
            nn.ReLU(True)
        )

        def block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(True)
            )

        self.net = nn.Sequential(
            block(1024, 512),  # 8
            block(512, 256),   # 16
            block(256, 128),   # 32
            block(128, 64),    # 64
            block(64, 32),     # 128
            block(32, 16),     # 256
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise, attrs):
        x = torch.cat([noise, attrs], dim=1)
        x = self.fc(x).view(-1, 1024, 4, 4)
        return self.net(x)

# ======================
# DISCRIMINATOR (Projection + SN)
# ======================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def sn_block(in_c, out_c):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.net = nn.Sequential(
            sn_block(3, 32),    # 128
            sn_block(32, 64),   # 64
            sn_block(64, 128),  # 32
            sn_block(128, 256), # 16
            sn_block(256, 512), # 8
            sn_block(512, 1024) # 4
        )

        self.fc = nn.utils.spectral_norm(nn.Linear(4*4*1024, 1))
        self.embed = nn.utils.spectral_norm(nn.Linear(ATTR_DIM, 4*4*1024))

    def forward(self, x, attrs):
        h = self.net(x)
        h = h.view(h.size(0), -1)

        out = self.fc(h)
        proj = torch.sum(self.embed(attrs) * h, dim=1, keepdim=True)

        return out + proj

# ======================
# EMA UPDATE
# ======================
def update_ema(ema_model, model):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(EMA_BETA).add_(p.data, alpha=1 - EMA_BETA)

# ======================
# INIT
# ======================
dataset = EyeDataset("final_eye_attributes_with_celeba.csv", "merged_eyes")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)
G_ema = Generator().to(DEVICE)
G_ema.load_state_dict(G.state_dict())

optimizer_G = optim.Adam(G.parameters(), lr=LR_G, betas=(0.0, 0.9))
optimizer_D = optim.Adam(D.parameters(), lr=LR_D, betas=(0.0, 0.9))

scaler = torch.cuda.amp.GradScaler()

# ======================
# TRAINING LOOP
# ======================
for epoch in range(EPOCHS):

    pbar = tqdm(loader)

    for real_imgs, attrs in pbar:

        real_imgs = real_imgs.to(DEVICE)
        attrs = attrs.to(DEVICE)
        b_size = real_imgs.size(0)

        # -------------------------
        # Train Discriminator
        # -------------------------
        optimizer_D.zero_grad()

        noise = torch.randn(b_size, NOISE_DIM, device=DEVICE)

        with torch.cuda.amp.autocast():
            fake_imgs = G(noise, attrs)

            real_out = D(real_imgs, attrs)
            fake_out = D(fake_imgs.detach(), attrs)

            d_loss = torch.mean(F.relu(1. - real_out)) + \
                     torch.mean(F.relu(1. + fake_out))

        scaler.scale(d_loss).backward()

        # R1 Regularization (every 16 steps)
        if np.random.randint(0, 16) == 0:
            real_imgs.requires_grad = True
            real_scores = D(real_imgs, attrs)
            grad_real = torch.autograd.grad(
                outputs=real_scores.sum(),
                inputs=real_imgs,
                create_graph=True
            )[0]
            r1_penalty = grad_real.pow(2).reshape(b_size, -1).sum(1).mean()
            scaler.scale(R1_GAMMA * r1_penalty).backward()

        scaler.step(optimizer_D)
        scaler.update()

        # -------------------------
        # Train Generator (2x)
        # -------------------------
        for _ in range(2):

            optimizer_G.zero_grad()
            noise = torch.randn(b_size, NOISE_DIM, device=DEVICE)

            with torch.cuda.amp.autocast():
                fake_imgs = G(noise, attrs)
                fake_out = D(fake_imgs, attrs)
                g_loss = -torch.mean(fake_out)

            scaler.scale(g_loss).backward()
            scaler.step(optimizer_G)
            scaler.update()

        update_ema(G_ema, G)

        pbar.set_description(
            f"Epoch {epoch+1} | D: {d_loss.item():.3f} | G: {g_loss.item():.3f}"
        )

    # -------------------------
    # SAVE SAMPLE + CHECKPOINT
    # -------------------------
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            test_noise = torch.randn(4, NOISE_DIM, device=DEVICE)
            test_attrs = torch.randint(0, 2, (4, ATTR_DIM), device=DEVICE).float()
            sample_imgs = G_ema(test_noise, test_attrs)

        vutils.save_image(
            sample_imgs,
            f"samples_256/epoch_{epoch+1}.png",
            normalize=True
        )

        torch.save({
            "G": G.state_dict(),
            "G_ema": G_ema.state_dict()
        }, f"checkpoints_256/g_epoch_{epoch+1}.pth")

    print(f"Epoch {epoch+1} completed.")