import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

########################################
# SETTINGS
########################################

CSV_PATH = r"D:\major_phase2\balanced_hair_dataset.csv"
IMG_DIR = "hair_sketch"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ATTR_COLS = [
    "Bald",
    "length_short","length_medium","length_long",
    "volume_thin","volume_normal","volume_thick",
    "Straight_Hair","Wavy_Hair"
]

########################################
# DATASET
########################################

class HairDataset(Dataset):
    def __init__(self,csv_path,img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]

        # condition map
        cond=[]
        for col in ATTR_COLS:
            cond.append(np.ones((IMG_SIZE,IMG_SIZE))*row[col])

        cond=torch.tensor(np.stack(cond,0)).float()

        # sketch image
        img_path=os.path.join(self.img_dir,row["image_id"])
        img=cv2.imread(img_path,0)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img=img/127.5 - 1
        img=torch.tensor(img).unsqueeze(0).float()

        return cond,img

########################################
# GENERATOR (U-Net)
########################################

def down(in_c,out_c):
    return nn.Sequential(
        nn.Conv2d(in_c,out_c,4,2,1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2)
    )

def up(in_c,out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c,out_c,4,2,1),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    )

class Generator(nn.Module):
    def __init__(self,in_c=9):
        super().__init__()

        self.d1=down(in_c,64)
        self.d2=down(64,128)
        self.d3=down(128,256)
        self.d4=down(256,512)

        self.u1=up(512,256)
        self.u2=up(512,128)
        self.u3=up(256,64)
        self.u4=nn.ConvTranspose2d(128,1,4,2,1)

        self.tanh=nn.Tanh()

    def forward(self,x):
        d1=self.d1(x)
        d2=self.d2(d1)
        d3=self.d3(d2)
        d4=self.d4(d3)

        u1=self.u1(d4)
        u1=torch.cat([u1,d3],1)

        u2=self.u2(u1)
        u2=torch.cat([u2,d2],1)

        u3=self.u3(u2)
        u3=torch.cat([u3,d1],1)

        return self.tanh(self.u4(u3))

########################################
# DISCRIMINATOR (PatchGAN)
########################################

class Discriminator(nn.Module):
    def __init__(self,in_c=10):
        super().__init__()

        self.net=nn.Sequential(
            nn.Conv2d(in_c,64,4,2,1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256,1,4,1,1)
        )

    def forward(self,x,y):
        return self.net(torch.cat([x,y],1))

########################################
# TRAINING
########################################

dataset=HairDataset(CSV_PATH,IMG_DIR)
loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

G=Generator().to(DEVICE)
D=Discriminator().to(DEVICE)

optG=torch.optim.Adam(G.parameters(),2e-4,betas=(0.5,0.999))
optD=torch.optim.Adam(D.parameters(),2e-4,betas=(0.5,0.999))

bce=nn.BCEWithLogitsLoss()
l1=nn.L1Loss()

os.makedirs("checkpoints1",exist_ok=True)

for epoch in range(EPOCHS):
    for cond,real in loader:
        cond,real=cond.to(DEVICE),real.to(DEVICE)

        ################################
        # Train Discriminator
        ################################
        fake=G(cond)

        D_real=D(cond,real)
        D_fake=D(cond,fake.detach())

        lossD=(bce(D_real,torch.ones_like(D_real)) +
               bce(D_fake,torch.zeros_like(D_fake)))/2

        optD.zero_grad()
        lossD.backward()
        optD.step()

        ################################
        # Train Generator
        ################################
        D_fake=D(cond,fake)
        lossG=bce(D_fake,torch.ones_like(D_fake)) + 100*l1(fake,real)

        optG.zero_grad()
        lossG.backward()
        optG.step()

    print(f"Epoch {epoch} | G:{lossG.item():.3f} D:{lossD.item():.3f}")

    torch.save({
        "G":G.state_dict(),
        "D":D.state_dict()
    },f"checkpoints1/pix2pix_{epoch}.pth")

print("✅ Training Complete!")
