import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import pandas as pd
from PIL import Image
import os

# --- 1. CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATTR_DIM = 6    
LATENT_DIM = 100  
IMG_SIZE = 128    
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.0002
CHECKPOINT_PATH = "forensic_eye_checkpoint.pth"

# Paths (Adjust to your local paths)
CSV_PATH = r'D:\major_phase2\final_cleaned.csv'
MERGED_DIR = r'D:\major_phase2\merged_forensic_targets'

# Weight for 'Bushy' (Index 1)
POS_WEIGHTS = torch.tensor([1.0, 5.3, 1.0, 1.0, 1.0, 1.0]).to(DEVICE)

# --- 2. ARCHITECTURE ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = IMG_SIZE // 16 
        self.l1 = nn.Sequential(nn.Linear(LATENT_DIM + ATTR_DIM, 512 * self.init_size**2))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, attrs):
        out = self.l1(torch.cat([z, attrs], dim=1))
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        return self.conv_blocks(out)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def block(in_f, out_f, bn=True):
            b = [nn.Conv2d(in_f, out_f, 3, 2, 1), nn.LeakyReLU(0.2), nn.Dropout2d(0.25)]
            if bn: b.append(nn.BatchNorm2d(out_f, 0.8))
            return b
        self.model = nn.Sequential(*block(1, 16, False), *block(16, 32), *block(32, 64), *block(64, 128))
        self.adv_layer = nn.Sequential(nn.Linear(128 * 8 * 8, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * 8 * 8, ATTR_DIM), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img).view(img.shape[0], -1)
        return self.adv_layer(out), self.aux_layer(out)

# --- 3. DATA LOADER ---
class ForensicDataset(Dataset):
    def __init__(self, csv, img_dir):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        image = Image.open(os.path.join(self.img_dir, img_name)).convert('L')
        attrs = torch.tensor(self.df.iloc[idx, 1:7].values.astype('float32'))
        return self.transform(image), attrs

# --- 4. MAIN TRAINING LOOP ---
def train():
    dataset = ForensicDataset(CSV_PATH, MERGED_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    netG = Generator().to(DEVICE)
    netD = Discriminator().to(DEVICE)

    optG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
    
    adv_loss = nn.BCELoss()
    aux_loss = nn.BCELoss(reduction='none')

    start_epoch = 0

    # --- CHECKPOINT LOADING ---
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH)
        netG.load_state_dict(checkpoint['G_state'])
        netD.load_state_dict(checkpoint['D_state'])
        optG.load_state_dict(checkpoint['G_opt'])
        optD.load_state_dict(checkpoint['D_opt'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        for i, (imgs, attrs) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            real_imgs, attrs = imgs.to(DEVICE), attrs.to(DEVICE)

            # Train Discriminator
            optD.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_imgs = netG(z, attrs)
            
            real_pred, real_aux = netD(real_imgs)
            fake_pred, _ = netD(fake_imgs.detach())

            d_loss = (adv_loss(real_pred, torch.ones(batch_size, 1).to(DEVICE)) + 
                      adv_loss(fake_pred, torch.zeros(batch_size, 1).to(DEVICE))) / 2
            d_aux = (aux_loss(real_aux, attrs) * POS_WEIGHTS).mean()
            (d_loss + d_aux).backward(); optD.step()

            # Train Generator
            optG.zero_grad()
            valid, pred_aux = netD(fake_imgs)
            g_loss = adv_loss(valid, torch.ones(batch_size, 1).to(DEVICE))
            g_aux = (aux_loss(pred_aux, attrs) * POS_WEIGHTS).mean()
            (g_loss + g_aux).backward(); optG.step()

        print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # --- SAVE CHECKPOINT ---
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'G_state': netG.state_dict(),
                'D_state': netD.state_dict(),
                'G_opt': optG.state_dict(),
                'D_opt': optD.state_dict()
            }
            torch.save(checkpoint, CHECKPOINT_PATH)
            
            # Validation Image: [Arched:0, Bushy:1, Narrow:0, Bags:0, Spects:0, Male:1, Young:0]
            val_attr = torch.tensor([[0, 1, 0, 0, 0, 1]], dtype=torch.float).to(DEVICE)
            val_z = torch.randn(1, LATENT_DIM).to(DEVICE)
            with torch.no_grad():
                sample = netG(val_z, val_attr)
                utils.save_image(sample, f"val_epoch_{epoch}.png", normalize=True)

if __name__ == "__main__":
    train()