import os
import torch
import torch.nn as nn
import numpy as np
import cv2

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints1/pix2pix_37.pth"  # change to your trained checkpoint

ATTR_COLS = [
    "Bald",
    "length_short", "length_medium", "length_long",
    "volume_thin", "volume_normal", "volume_thick",
    "Straight_Hair", "Wavy_Hair"
]

# -----------------------------
# GENERATOR (same as training)
# -----------------------------
def down(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 4, 2, 1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2)
    )

def up(in_c, out_c):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    )

class Generator(nn.Module):
    def __init__(self, in_c=9):
        super().__init__()
        self.d1 = down(in_c, 64)
        self.d2 = down(64, 128)
        self.d3 = down(128, 256)
        self.d4 = down(256, 512)

        self.u1 = up(512, 256)
        self.u2 = up(512, 128)
        self.u3 = up(256, 64)
        self.u4 = nn.ConvTranspose2d(128, 1, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        u1 = self.u1(d4)
        u1 = torch.cat([u1, d3], 1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d2], 1)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, d1], 1)

        return self.tanh(self.u4(u3))

# -----------------------------
# LOAD GENERATOR
# -----------------------------
G = Generator().to(DEVICE)
checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
G.load_state_dict(checkpoint["G"])
G.eval()

# -----------------------------
# FUNCTION: text -> attribute tensor
# -----------------------------
def text_to_tensor(text):
    """
    text: string like "Bald, length_short, volume_thick, Wavy_Hair"
    returns: (1, 9, IMG_SIZE, IMG_SIZE) torch tensor
    """
    attr = np.zeros(len(ATTR_COLS), dtype=np.float32)
    text_attrs = [t.strip() for t in text.split(",")]
    for i, col in enumerate(ATTR_COLS):
        if col in text_attrs:
            attr[i] = 1.0

    # repeat to image size
    attr_map = np.stack([np.ones((IMG_SIZE, IMG_SIZE)) * a for a in attr], 0)
    return torch.tensor(attr_map).unsqueeze(0).float()  # shape: 1x9x128x128

# -----------------------------
# FUNCTION: generate and save image
# -----------------------------
def generate_hair(text, save_path="generated_hair1.png"):
    with torch.no_grad():
        cond = text_to_tensor(text).to(DEVICE)
        fake = G(cond)
        fake_img = fake.squeeze().cpu().numpy()
        fake_img = ((fake_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
        cv2.imwrite(save_path, fake_img)
        print(f"✅ Hair image saved at: {save_path}")

# -----------------------------
# EXAMPLE USAGE
# -----------------------------
if __name__ == "__main__":
    print("Enter hair attributes separated by commas (e.g., Bald, length_short, volume_thick, Wavy_Hair)")
    user_input = input("Your input: ")
    generate_hair(user_input)
