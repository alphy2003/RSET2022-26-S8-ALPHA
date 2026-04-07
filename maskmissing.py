import pandas as pd
import os

# Path to CSV-style CelebA attributes
ATTR_PATH = r"D:\Dataset\list_attr_celeba.csv"

# Folder containing your 110,907 masked images
MASKED_FOLDER = r"D:\major_phase2\masked_inputs"

# Load CSV directly
df = pd.read_csv(ATTR_PATH)

# Convert -1/1 to 0/1
df = df.replace(-1, 0)

# Normalize filenames
masked_files = {f.replace("_mask.png", ".jpg") for f in os.listdir(MASKED_FOLDER)}

# Filter attributes for images that have a mask
df_final = df[df["image_id"].isin(masked_files)]

print("Total final images =", len(df_final))

# Attributes you want
ATTRS = [
    "Male",
    "Young",
    "Narrow_Eyes",
    "Smiling",
    "Bushy_Eyebrows",
    "Arched_Eyebrows",
    "Heavy_Makeup"
]

print("\n=== Attribute Distribution in Final Dataset ===")
for attr in ATTRS:
    count = df_final[attr].sum()
    pct = (count / len(df_final)) * 100
    print(f"{attr:18}: {count} samples ({pct:.2f}%)")
