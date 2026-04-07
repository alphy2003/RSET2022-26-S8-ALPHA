import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

############################################
# CONFIG
############################################

CSV_PATH = r"D:\major_phase2\balanced_hair_dataset.csv"
IMG_DIR = r"D:\major_phase2\hair_sketch"
SAVE_DIR = "checkpoints_hair"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 100
ATTR_DIM = 9   # number of attributes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################################
# DATASET
############################################

class HairDataset(Dataset):
    def __init__(self,csv,img_dir):
        self.df = pd.read_csv(csv)
        self.dir = img_dir

        self.attr_cols = [
            "Bald",
            "Straight_Hair","Wavy_Hair",
            "length_short","length_medium","length_long",
            "volume_thin","volume_normal","volume_thick"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]

        img = cv2.imread(os.path.join(self.dir,row["image_id"]),0)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img = img/127.5 - 1
        img = torch.tensor(img).unsqueeze(0).float()

        attrs = torch.tensor(row[self.attr_cols].values.astype(np.float32))

        return img,attrs

############################################
# HANDLE IMBALANCE
############################################

df = pd.read_csv(CSV_PATH)

# oversample short hair
weights = []
for _,r in df.iterrows():
    if r["length_short"]==1:
        weights.append(3.0)
    else:
        weights.append(1.0)

sampler = WeightedRandomSampler(weights,len(weights))

dataset = HairDataset(CSV_PATH,IMG_DIR)
loader = DataLoader(dataset,batch_size=BATCH_SIZE,sampler=sampler)

############################################
# MODELS
############################################

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM+ATTR_DIM,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,IMG_SIZE*IMG_SIZE),
            nn.Tanh()
        )

    def forward(self,z,attrs):
        x = torch.cat([z,attrs],1)
        img = self.net(x)
        return img.view(-1,1,IMG_SIZE,IMG_SIZE)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_SIZE*IMG_SIZE+ATTR_DIM,512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,img,attrs):
        img = img.view(img.size(0),-1)
        x = torch.cat([img,attrs],1)
        return self.net(x)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

############################################
# TRAIN SETUP
############################################

criterion = nn.BCELoss()
optG = torch.optim.Adam(G.parameters(),0.0002)
optD = torch.optim.Adam(D.parameters(),0.0002)

best_loss = 999

############################################
# TRAIN LOOP
############################################

for epoch in range(EPOCHS):

    for real,attrs in loader:

        real,attrs = real.to(DEVICE),attrs.to(DEVICE)
        bs = real.size(0)

        real_label = torch.ones(bs,1).to(DEVICE)*0.9  # label smoothing
        fake_label = torch.zeros(bs,1).to(DEVICE)

        ################################
        # TRAIN D
        ################################
        z = torch.randn(bs,LATENT_DIM).to(DEVICE)
        fake = G(z,attrs)

        lossD = (
            criterion(D(real,attrs),real_label)+
            criterion(D(fake.detach(),attrs),fake_label)
        )/2

        optD.zero_grad()
        lossD.backward()
        optD.step()

        ################################
        # TRAIN G
        ################################
        lossG = criterion(D(fake,attrs),real_label)

        optG.zero_grad()
        lossG.backward()
        optG.step()

    print(f"Epoch {epoch} | D {lossD.item():.3f} | G {lossG.item():.3f}")

    ################################
    # CHECKPOINT
    ################################
    if (epoch+1)%5==0:
        torch.save({
            "G":G.state_dict(),
            "D":D.state_dict(),
            "optG":optG.state_dict(),
            "optD":optD.state_dict(),
            "epoch":epoch
        },f"{SAVE_DIR}/ckpt_{epoch+1}.pth")

        print("✅ checkpoint saved")

    ################################
    # SAVE BEST
    ################################
    if lossG.item()<best_loss:
        best_loss = lossG.item()
        torch.save(G.state_dict(),f"{SAVE_DIR}/best_generator.pth")

print("🎉 Training complete!")
