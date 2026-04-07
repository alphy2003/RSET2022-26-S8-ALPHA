import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

##################################
# SETTINGS
##################################
CSV_PATH = r"D:\major_phase2\balanced_hair_dataset.csv"
IMG_DIR = r"D:\major_phase2\hair_sketch"
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ATTR_COLS = [
    "Bald","length_short","length_medium","length_long",
    "volume_thin","volume_normal","volume_thick",
    "Straight_Hair","Wavy_Hair"
]

##################################
# DATASET
##################################
class HairDataset(Dataset):
    def __init__(self,csv,img_dir):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]

        attr = torch.tensor(row[ATTR_COLS].values.astype(np.float32))

        path = os.path.join(self.img_dir,row["image_id"])
        img = cv2.imread(path,0)

        if img is None:
            img = np.ones((IMG_SIZE,IMG_SIZE))*255

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        img = img/127.5 - 1
        img = torch.tensor(img).unsqueeze(0).float()

        return attr,img

##################################
# GENERATOR
##################################
class Generator(nn.Module):
    def __init__(self,attr_dim=9):
        super().__init__()

        self.attr_fc = nn.Linear(attr_dim,128)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256,256,4,1,0),
            nn.InstanceNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256,128,4,2,1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128,64,4,2,1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64,32,4,2,1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32,16,4,2,1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16,1,4,2,1),
            nn.Tanh()
        )

    def forward(self,z,attr):
        a = self.attr_fc(attr).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z,a],1)
        return self.net(x)

##################################
# DISCRIMINATOR (with features)
##################################
class Discriminator(nn.Module):
    def __init__(self,attr_dim=9):
        super().__init__()

        self.attr_fc = nn.Linear(attr_dim,IMG_SIZE*IMG_SIZE)

        self.conv1 = nn.Conv2d(2,64,4,2,1)
        self.conv2 = nn.Conv2d(64,128,4,2,1)
        self.conv3 = nn.Conv2d(128,256,4,2,1)
        self.conv4 = nn.Conv2d(256,1,4,1,0)

        self.norm2 = nn.InstanceNorm2d(128)
        self.norm3 = nn.InstanceNorm2d(256)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self,img,attr,return_feat=False):
        a = self.attr_fc(attr).view(-1,1,IMG_SIZE,IMG_SIZE)
        x = torch.cat([img,a],1)

        f1 = self.lrelu(self.conv1(x))
        f2 = self.lrelu(self.norm2(self.conv2(f1)))
        f3 = self.lrelu(self.norm3(self.conv3(f2)))
        out = self.conv4(f3)

        if return_feat:
            return out,[f1,f2,f3]
        return out

##################################
# TRAINING
##################################
dataset = HairDataset(CSV_PATH,IMG_DIR)
loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

optG = torch.optim.Adam(G.parameters(),2e-4,betas=(0.5,0.999))
optD = torch.optim.Adam(D.parameters(),1e-4,betas=(0.5,0.999))

bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()

os.makedirs("samples",exist_ok=True)

for epoch in range(EPOCHS):
    for attr,real in loader:

        attr,real = attr.to(DEVICE),real.to(DEVICE)
        bs = real.size(0)

        ##################################
        # Train D
        ##################################
        z = torch.randn(bs,128,1,1).to(DEVICE)
        fake = G(z,attr)

        d_real = D(real,attr)
        d_fake = D(fake.detach(),attr)

        real_lbl = torch.ones_like(d_real)*0.9
        fake_lbl = torch.zeros_like(d_fake)

        lossD = (bce(d_real,real_lbl)+bce(d_fake,fake_lbl))/2

        optD.zero_grad()
        lossD.backward()
        optD.step()

        ##################################
        # Train G
        ##################################
        z = torch.randn(bs,128,1,1).to(DEVICE)
        fake = G(z,attr)

        d_fake,feat_fake = D(fake,attr,True)
        _,feat_real = D(real,attr,True)

        adv_loss = bce(d_fake,torch.ones_like(d_fake))

        fm_loss = 0
        for fr,ff in zip(feat_real,feat_fake):
            fm_loss += l1(ff,fr.detach())

        lossG = adv_loss + 10*fm_loss

        optG.zero_grad()
        lossG.backward()
        optG.step()

    print(f"Epoch {epoch} | G {lossG.item():.3f} | D {lossD.item():.3f}")

    if epoch%5==0:
        save_image(fake[:4]*0.5+0.5,f"samples/epoch_{epoch}.png")

    torch.save(G.state_dict(),f"hair_gen_{epoch}.pth")

print("✅ Done!")
