import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Parameters
# =========================

IMAGE_SIZE = 256
BATCH_SIZE = 16
LATENT_DIM = 100
ATTR_DIM = 3
EPOCHS = 200

CSV_PATH = "face_shape_onehot.csv"
IMAGE_FOLDER = r"D:\major_phase2\face_outlines"

CHECKPOINT_PATH = "outline_checkpoint1.pth"

# =========================
# Dataset
# =========================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

class OutlineDataset(Dataset):

    def __init__(self,csv_file,img_folder,transform=None):

        self.data = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        img_name = self.data.iloc[idx]['image_id']
        img_path = os.path.join(self.img_folder,img_name)

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        attributes = torch.tensor(
            self.data.iloc[idx][1:].values.astype("float32")
        )

        return image, attributes

dataset = OutlineDataset(CSV_PATH,IMAGE_FOLDER,transform)
loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

# =========================
# Generator
# =========================

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(LATENT_DIM + ATTR_DIM,1024),
            nn.ReLU(),
            nn.Linear(1024,256*16*16),
            nn.ReLU()
        )

        self.conv = nn.Sequential(

            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64,32,4,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32,1,4,2,1),
            nn.Tanh()
        )

    def forward(self,z,attr):

        x = torch.cat([z,attr],dim=1)
        x = self.fc(x)
        x = x.view(-1,256,16,16)
        x = self.conv(x)

        return x

# =========================
# Discriminator
# =========================

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(1,32,4,2,1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256*16*16 + ATTR_DIM,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )

    def forward(self,img,attr):

        x = self.conv(img)
        x = x.view(x.size(0),-1)

        x = torch.cat([x,attr],dim=1)

        x = self.fc(x)

        return x

# =========================
# Initialize Models
# =========================

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

optimizerG = optim.Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD = optim.Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))

start_epoch = 0

# =========================
# Load Checkpoint if exists
# =========================

if os.path.exists(CHECKPOINT_PATH):

    checkpoint = torch.load(CHECKPOINT_PATH)

    G.load_state_dict(checkpoint['G'])
    D.load_state_dict(checkpoint['D'])

    optimizerG.load_state_dict(checkpoint['optimizerG'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])

    start_epoch = checkpoint['epoch']

    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")

# =========================
# Training
# =========================

for epoch in range(start_epoch,EPOCHS):

    for real_imgs,attrs in loader:

        real_imgs = real_imgs.to(device)
        attrs = attrs.to(device)

        batch_size = real_imgs.size(0)

        real_label = torch.ones(batch_size,1).to(device)
        fake_label = torch.zeros(batch_size,1).to(device)

        # ---- Train Discriminator ----

        z = torch.randn(batch_size,LATENT_DIM).to(device)

        fake_imgs = G(z,attrs)

        real_loss = criterion(D(real_imgs,attrs),real_label)
        fake_loss = criterion(D(fake_imgs.detach(),attrs),fake_label)

        d_loss = real_loss + fake_loss

        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        # ---- Train Generator ----

        output = D(fake_imgs,attrs)

        g_loss = criterion(output,real_label)

        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

    print(f"Epoch {epoch+1}/{EPOCHS}  D_loss:{d_loss.item():.4f}  G_loss:{g_loss.item():.4f}")

    # =========================
    # Save Checkpoint
    # =========================

    torch.save({

        'epoch': epoch+1,
        'G': G.state_dict(),
        'D': D.state_dict(),
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict()

    }, CHECKPOINT_PATH)

    print("Checkpoint saved!")

# =========================
# Save Final Generator
# =========================

torch.save(G.state_dict(),"outline_generator.pth")

print("Training Completed!")