import torch
from torchvision.utils import save_image
import torch.nn as nn

# =============================
# Configuration
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
num_attrs = 6
generator_checkpoint = "checkpoints/generator_epoch_70.pth"

# =============================
# Generator Definition
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
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(True),

            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, labels), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 8)
        img = self.conv(x)
        return img

# =============================
# Load Generator
# =============================
generator = Generator().to(device)
generator.load_state_dict(torch.load(generator_checkpoint, map_location=device))
generator.eval()

# =============================
# Function to get attribute input
# =============================
def get_attributes():
    print("Enter 6 numerical nose attributes (values between 0 and 1).")
    attrs = []
    for i in range(1, num_attrs+1):
        while True:
            try:
                val = float(input(f"Attribute {i}: "))
                if 0.0 <= val <= 1.0:
                    attrs.append(val)
                    break
                else:
                    print("Enter a value between 0 and 1.")
            except:
                print("Invalid input, enter a number.")
    return torch.tensor([attrs], dtype=torch.float32).to(device)

# =============================
# Generate Nose Image
# =============================
attributes = get_attributes()

# Generate a random latent vector
z = torch.randn(1, latent_dim).to(device)

with torch.no_grad():
    generated_img = generator(z, attributes)

# Save output
save_image(generated_img, "generated_nose.png", normalize=True)
print("Nose image generated and saved as 'generated_nose.png'.")