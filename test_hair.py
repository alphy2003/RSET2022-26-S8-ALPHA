import torch
import torch.nn as nn
import numpy as np
import cv2
import os

##################################
# SETTINGS
##################################
MODEL_PATH = "hair_gen_new140.pth"   # change to your best epoch
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

##################################
# GENERATOR (EXACT SAME AS TRAINING)
##################################
class Generator(nn.Module):
    def __init__(self, attr_dim=9, noise_dim=128):
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
# LOAD MODEL
##################################
G = Generator().to(DEVICE)
G.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
G.eval()

print("✅ Model Loaded Successfully")

##################################
# USER INPUT FUNCTION
##################################
def get_user_input():

    print("\nEnter Hair Attributes (1 or 0)\n")

    bald = int(input("Bald (1/0): "))
    length_short = int(input("Short Length (1/0): "))
    length_medium = int(input("Medium Length (1/0): "))
    length_long = int(input("Long Length (1/0): "))
    volume_thin = int(input("Thin Volume (1/0): "))
    volume_normal = int(input("Normal Volume (1/0): "))
    volume_thick = int(input("Thick Volume (1/0): "))
    straight = int(input("Straight Hair (1/0): "))
    wavy = int(input("Wavy Hair (1/0): "))

    attr = torch.tensor([
        bald,length_short,length_medium,length_long,
        volume_thin,volume_normal,volume_thick,
        straight,wavy
    ], dtype=torch.float32).unsqueeze(0)

    return attr

##################################
# GENERATE IMAGE
##################################
def generate_hair():

    attr = get_user_input().to(DEVICE)

    z = torch.randn(1,128,1,1).to(DEVICE)

    with torch.no_grad():
        fake = G(z,attr)

    img = fake.squeeze().cpu().numpy()
    img = (img + 1) / 2
    img = (img * 255).astype(np.uint8)

    os.makedirs("generated", exist_ok=True)
    save_path = "generated/generated_hair.png"
    cv2.imwrite(save_path,img)

    print(f"\n✅ Hair Image Saved at: {save_path}")

    cv2.imshow("Generated Hair", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

##################################
# RUN
##################################
if __name__ == "__main__":
    generate_hair()