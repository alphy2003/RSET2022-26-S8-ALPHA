import torch
import torch.nn as nn
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
num_attrs = 3

# =============================
# Generator Architecture
# =============================

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_attrs, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 256 * 8 * 8),
            nn.ReLU(True),
        )

        self.conv = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,128,3,1,1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,1,1),
            nn.ReLU(True),

            nn.Conv2d(64,1,3,1,1),
            nn.Tanh()
        )

    def forward(self, noise, labels):

        x = torch.cat((noise, labels), dim=1)

        x = self.fc(x)

        x = x.view(x.size(0),256,8,8)

        img = self.conv(x)

        return img


# =============================
# Load Trained Model
# =============================

generator = Generator().to(device)

generator.load_state_dict(torch.load(
    "outline_checkpoints/generator_epoch_3.pth",
    map_location=device
))

generator.eval()

# =============================
# Ask User Input
# =============================

print("Select Face Shape")
print("1 - Oval")
print("2 - Round")
print("3 - Square")

choice = input("Enter your choice: ").lower()

if choice == "oval" or choice == "1":
    label = [1,0,0]
elif choice == "round" or choice == "2":
    label = [0,1,0]
elif choice == "square" or choice == "3":
    label = [0,0,1]
else:
    print("Invalid input. Please enter oval, round, or square.")
    exit()

label = torch.tensor([label]).float().to(device)

# =============================
# Generate Outline
# =============================

noise = torch.randn(1, latent_dim).to(device)

with torch.no_grad():
    generated_img = generator(noise, label)

save_image(generated_img, "generated_outline.png", normalize=True)

print("Outline generated and saved as generated_outline.png")