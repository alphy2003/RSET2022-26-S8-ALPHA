import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

# =============================
# Configuration
# =============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "updated_nose_attributes.csv"
image_folder = r"D:\major_phase2\nose_sharpened"

latent_dim = 100
num_attrs = 6
img_size = 64
channels = 1
batch_size = 64
epochs = 100
lr = 0.0002
lambda_gp = 10
critic_steps = 5

os.makedirs("samples", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# =============================
# Dataset
# =============================

class NoseDataset(Dataset):
    def __init__(self, csv_file, img_folder):
        self.df = pd.read_csv(csv_file)
        self.img_folder = img_folder

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row["image_id"])

        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        labels = torch.tensor(row[1:].values.astype(np.float32))

        return image, labels

# =============================
# Generator
# =============================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_attrs, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 256 * 8 * 8),
            nn.ReLU(True),
        )

        self.conv = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.Upsample(scale_factor=2),   # 8 → 16
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),   # 16 → 32
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),   # 32 → 64  ✅ FIX
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(True),

            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 8)
        img = self.conv(x)
        return img

# =============================
# Discriminator (Critic)
# =============================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1)
        )

    def forward(self, img, labels):
        label_map = labels.mean(dim=1, keepdim=True)
        label_map = label_map.view(label_map.size(0), 1, 1, 1)
        label_map = label_map.expand(-1, 1, img_size, img_size)

        x = torch.cat((img, label_map), dim=1)
        return self.model(x)

# =============================
# Gradient Penalty
# =============================

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    d_interpolates = D(interpolates, labels)

    fake = torch.ones(d_interpolates.size()).to(device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# =============================
# Training Setup
# =============================

dataset = NoseDataset(csv_path, image_folder)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# =============================
# Training Loop
# =============================

for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        imgs = imgs.to(device)
        labels = labels.to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------

        for _ in range(critic_steps):
            optimizer_D.zero_grad()

            z = torch.randn(imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(z, labels)

            real_validity = discriminator(imgs, labels)
            fake_validity = discriminator(fake_imgs.detach(), labels)

            gp = compute_gradient_penalty(discriminator, imgs.data, fake_imgs.data, labels)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

            d_loss.backward()
            optimizer_D.step()

        # ---------------------
        # Train Generator
        # ---------------------

        optimizer_G.zero_grad()

        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z, labels)

        fake_validity = discriminator(gen_imgs, labels)
        g_loss = -torch.mean(fake_validity)

        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

    # Save generated samples
    save_image(gen_imgs.data[:25], f"samples/epoch_{epoch+1}.png", nrow=5, normalize=True)

    # Save checkpoints
    torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch+1}.pth")
    torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch+1}.pth")

print("Training Completed!")
