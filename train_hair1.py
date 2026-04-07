import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.utils import save_image

##################################
# SETTINGS
##################################
CSV_PATH = r"D:\major_phase2\balanced_hair_dataset.csv"
IMG_DIR = r"D:\major_phase2\hair_sketch"
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 180
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
# GENERATOR (UPSAMPLE VERSION)
##################################
class Generator(nn.Module):
    def __init__(self,attr_dim=9,noise_dim=128):
        super().__init__()

        self.attr_fc = nn.Linear(attr_dim,128)

        self.init = nn.Sequential(
            nn.ConvTranspose2d(256,512,4,1,0),
            nn.ReLU()
        )

        def block(in_c,out_c):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_c,out_c,3,padding=1),
                nn.InstanceNorm2d(out_c),
                nn.ReLU()
            )

        self.net = nn.Sequential(
            block(512,256),
            block(256,128),
            block(128,64),
            block(64,32),
            block(32,16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16,1,3,padding=1),
            nn.Tanh()
        )

    def forward(self,z,attr):
        a = self.attr_fc(attr).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z,a],1)
        x = self.init(x)
        return self.net(x)

##################################
# PROJECTION DISCRIMINATOR
##################################
class Discriminator(nn.Module):
    def __init__(self,attr_dim=9):
        super().__init__()

        def block(in_c,out_c):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_c,out_c,4,2,1)),
                nn.LeakyReLU(0.2)
            )

        self.conv = nn.Sequential(
            block(1,64),
            block(64,128),
            block(128,256),
            block(256,512),
        )

        self.final = nn.utils.spectral_norm(nn.Conv2d(512,1,4,1,0))
        self.embed = nn.Linear(attr_dim,512)

    def forward(self,img,attr):
        h = self.conv(img)
        out = self.final(h)
        out = torch.mean(out, dim=[2,3])

        pooled = torch.mean(h,dim=[2,3])
        proj = torch.sum(self.embed(attr)*pooled,dim=1,keepdim=True)

        return out + proj

##################################
# LOAD DATA WITH BALANCED SAMPLER
##################################
dataset = HairDataset(CSV_PATH,IMG_DIR)
df = pd.read_csv(CSV_PATH)

bald_counts = df["Bald"].value_counts()
weight0 = 1.0 / bald_counts[0]
weight1 = 1.0 / bald_counts[1]

weights = df["Bald"].apply(lambda x: weight1 if x==1 else weight0)

sampler = WeightedRandomSampler(
    torch.DoubleTensor(weights.values),
    len(weights),
    replacement=True
)

loader = DataLoader(dataset,batch_size=BATCH_SIZE,sampler=sampler)

##################################
# MODELS
##################################
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

optG = torch.optim.Adam(G.parameters(),1e-4,betas=(0.5,0.999))
optD = torch.optim.Adam(D.parameters(),4e-4,betas=(0.5,0.999))

bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()

os.makedirs("samples",exist_ok=True)

##################################
# TRAINING LOOP
##################################
for epoch in range(EPOCHS):
    for attr,real in loader:

        attr,real = attr.to(DEVICE),real.to(DEVICE)
        bs = real.size(0)

        real = real + 0.03*torch.randn_like(real)
        real = torch.clamp(real,-1,1)

        ##################################
        # TRAIN DISCRIMINATOR
        ##################################
        z = torch.randn(bs,128,1,1).to(DEVICE)
        fake = G(z,attr)

        real.requires_grad_(True)

        d_real = D(real,attr)
        d_fake = D(fake.detach(),attr)

        real_lbl = torch.ones_like(d_real)*0.9
        fake_lbl = torch.zeros_like(d_fake)

        adv_loss = (bce(d_real,real_lbl)+bce(d_fake,fake_lbl))/2

        grad_real = torch.autograd.grad(
            outputs=d_real.sum(),
            inputs=real,
            create_graph=True
        )[0]

        r1_penalty = grad_real.pow(2).reshape(bs,-1).sum(1).mean()

        lossD = adv_loss + 5*r1_penalty

        optD.zero_grad()
        lossD.backward()
        optD.step()

        ##################################
        # TRAIN GENERATOR
        ##################################
        z = torch.randn(bs,128,1,1).to(DEVICE)
        fake = G(z,attr)

        d_fake = D(fake,attr)
        adv_loss_g = bce(d_fake,torch.ones_like(d_fake))

        recon_loss = l1(fake,real)

        lossG = adv_loss_g + 2*recon_loss

        optG.zero_grad()
        lossG.backward()
        optG.step()

    print(f"Epoch {epoch} | G {lossG.item():.3f} | D {lossD.item():.3f}")

    if epoch % 1 == 0:
        save_image(fake[:4]*0.5+0.5,f"samples/epoch_{epoch}.png")

    torch.save(G.state_dict(),f"hair_gen_new1{epoch}.pth")

print("✅ High Quality Training Complete")