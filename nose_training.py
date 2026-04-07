import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ================== CONFIG ==================
CSV_PATH = r"D:\major_phase2\nose_final_train.csv"
IMG_DIR = r"D:\major_phase2\nose_clean"

SAVE_ROOT = r"D:\major_phase2\gan_results"
CKPT_DIR = os.path.join(SAVE_ROOT,"checkpoints_nose")
SAMPLE_DIR = os.path.join(SAVE_ROOT,"samples1")

os.makedirs(CKPT_DIR,exist_ok=True)
os.makedirs(SAMPLE_DIR,exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 80
Z_DIM = 100
IMG_SIZE = 64
ATTR_DIM = 4
LR = 0.0002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== DATASET ==================
class NoseDataset(Dataset):
    def __init__(self,csv_path,img_dir):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]

        img = cv2.imread(
            os.path.join(self.img_dir,row['image_id']),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            img = np.ones((64,64))*255

        img = cv2.resize(img,(64,64))
        img = img/127.5 - 1
        img = torch.tensor(img).float().unsqueeze(0)

        # flip augmentation
        if torch.rand(1) > 0.5:
            img = torch.flip(img,[2])

        attrs = torch.tensor([
            row['Big_Nose'],
            row['Pointy_Nose'],
            row['Male'],
            row['Young']
        ]).float()

        return img,attrs

dataset = NoseDataset(CSV_PATH,IMG_DIR)
loader = DataLoader(dataset,BATCH_SIZE,shuffle=True)

# ================== GENERATOR ==================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Linear(ATTR_DIM,16)
        self.init = nn.Linear(Z_DIM+16,512*4*4)

        self.net = nn.Sequential(
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64,1,4,2,1),
            nn.Tanh()
        )

    def forward(self,z,labels):
        c = self.label_emb(labels)
        x = torch.cat([z,c],1)
        x = self.init(x).view(-1,512,4,4)
        return self.net(x)

# ================== DISCRIMINATOR ==================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Linear(ATTR_DIM,64*64)

        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(2,64,4,2,1)),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(64,128,4,2,1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(128,256,4,2,1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.utils.spectral_norm(nn.Conv2d(256,512,4,2,1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        self.fc = nn.Linear(512*4*4,1)

    def forward(self,img,labels):
        label_map = self.label_emb(labels).view(-1,1,64,64)
        x = torch.cat([img,label_map],1)

        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)  # BCEWithLogitsLoss

# ================== INIT ==================
G = Generator().to(device)
D = Discriminator().to(device)

opt_G = optim.Adam(G.parameters(),LR,betas=(0.5,0.999))
opt_D = optim.Adam(D.parameters(),LR,betas=(0.5,0.999))

criterion = nn.BCEWithLogitsLoss()

# ================== RANDOM ATTR ==================
def random_attrs(n):
    return torch.randint(0,2,(n,4)).float().to(device)

# ================== TRAIN ==================
for epoch in range(EPOCHS):

    for imgs,attrs in tqdm(loader):

        imgs,attrs = imgs.to(device),attrs.to(device)
        bs = imgs.size(0)

        imgs += 0.05*torch.randn_like(imgs)  # instance noise

        real = torch.ones(bs,1).to(device)*0.9
        fake = torch.zeros(bs,1).to(device)

        # ---- Train D ----
        z = torch.randn(bs,Z_DIM).to(device)
        fake_imgs = G(z,attrs)

        d_real = D(imgs,attrs)
        d_fake = D(fake_imgs.detach(),attrs)

        loss_D = criterion(d_real,real) + \
                 criterion(d_fake,fake)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # ---- Train G ----
        d_fake = D(fake_imgs,attrs)
        loss_G = criterion(d_fake,real)

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch} | D {loss_D.item():.3f} | G {loss_G.item():.3f}")

    # ===== SAVE CHECKPOINT =====
    torch.save({
        "G":G.state_dict(),
        "D":D.state_dict(),
        "opt_G":opt_G.state_dict(),
        "opt_D":opt_D.state_dict(),
    }, f"{CKPT_DIR}/epoch_{epoch}.pth")

    # ===== GENERATE SAMPLES =====
    with torch.no_grad():
        z = torch.randn(5,Z_DIM).to(device)
        attrs_rand = random_attrs(5)
        samples = G(z,attrs_rand)

        for i,img in enumerate(samples):
            img = (img.cpu().numpy()+1)*127.5
            attr_txt = "_".join(map(str,attrs_rand[i].int().tolist()))
            cv2.imwrite(
                f"{SAMPLE_DIR}/epoch{epoch}_{attr_txt}.png",
                img[0]
            )

print("🎉 Training Finished!")
