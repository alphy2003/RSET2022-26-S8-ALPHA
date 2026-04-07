import torch
import torch.nn as nn
import numpy as np
import cv2

########################################
# SETTINGS
########################################
IMG_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 👉 CHANGE TO YOUR CHECKPOINT
CHECKPOINT = "generator_v2_epoch_199.pth"

ATTR_COLS = [
    "Bald",
    "length_short","length_medium","length_long",
    "volume_thin","volume_normal","volume_thick",
    "Straight_Hair","Wavy_Hair"
]

########################################
# GENERATOR (EXACT COPY FROM TRAINING)
########################################
def conv_block(in_c, out_c, down=True):
    if down:
        return nn.Sequential(
            nn.Conv2d(in_c,out_c,4,2,1,bias=False),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c,out_c,4,2,1,bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

class Generator(nn.Module):
    def __init__(self, attr_dim=9):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1,64,4,2,1),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = conv_block(64,128)
        self.enc3 = conv_block(128,256)
        self.enc4 = conv_block(256,512)

        self.attr_mlp = nn.Sequential(
            nn.Linear(attr_dim,512),
            nn.ReLU()
        )

        self.dec1 = conv_block(1024,256,down=False)
        self.dec2 = conv_block(512,128,down=False)
        self.dec3 = conv_block(256,64,down=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128,1,4,2,1),
            nn.Tanh()
        )

    def forward(self, noise, attr):
        e1 = self.enc1(noise)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        a = self.attr_mlp(attr).view(-1,512,1,1)
        a = a.expand(-1,-1,e4.size(2),e4.size(3))

        bottleneck = torch.cat([e4,a],1)

        d1 = self.dec1(bottleneck)
        d2 = self.dec2(torch.cat([d1,e3],1))
        d3 = self.dec3(torch.cat([d2,e2],1))

        return self.final(torch.cat([d3,e1],1))

########################################
# LOAD MODEL
########################################
G = Generator().to(DEVICE)

G.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
G.eval()

print("✅ Model Loaded Successfully!")

########################################
# TEXT → ATTRIBUTE VECTOR
########################################
def text_to_attr(text):
    attrs = text.split(",")
    attrs = [a.strip() for a in attrs]

    vec = np.zeros(len(ATTR_COLS), dtype=np.float32)

    for i, col in enumerate(ATTR_COLS):
        if col in attrs:
            vec[i] = 1.0

    return torch.tensor(vec).unsqueeze(0)

########################################
# GENERATE FUNCTION
########################################
def generate(text):

    attr = text_to_attr(text).to(DEVICE)

    # random noise seed
    noise = torch.randn(1,1,IMG_SIZE,IMG_SIZE).to(DEVICE)

    with torch.no_grad():
        fake = G(noise, attr)

    img = fake.squeeze().cpu().numpy()

    # [-1,1] → [0,255]
    img = ((img + 1) * 127.5).clip(0,255).astype(np.uint8)

    cv2.imwrite("generated.png", img)

    print("✅ Image saved as generated.png")

########################################
# USER INPUT LOOP
########################################
print("\nAvailable attributes:")
print(ATTR_COLS)

while True:

    text = input("\nEnter attributes (comma separated) or 'q': ")

    if text.lower() == "q":
        break

    generate(text)
