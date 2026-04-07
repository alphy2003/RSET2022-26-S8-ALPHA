import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# ============================
# CONFIG
# ============================
ATTR_FILE = r"D:\Dataset\list_attr_celeba.csv"
IMG_DIR   = r"D:\Dataset\img_align_celeba\img_align_celeba"
OUT_DIR   = r"D:/major_phase2/output_soft_balanced"

TARGET_PER_CLASS = 6666      # ~20000 total (3 x 6666)

PRIMARY_CLASSES = {
    "Bushy":  "Bushy_Eyebrows",
    "Arched": "Arched_Eyebrows",
    "Makeup": "Heavy_Makeup"
}

SECONDARY_SOFT_BALANCE = [
    "Male",
    "Young",
    "Smiling",
    "Narrow_Eyes",
]

HARDCODE_FILTERS = {
    "Eyeglasses": -1,     # remove eyeglasses
    "Blurry": -1          # remove blurry images
}

# Create output directory
os.makedirs(OUT_DIR, exist_ok=True)

# ============================
# LOAD ATTRIBUTES
# ============================
print("📥 Loading CelebA attributes ...")
df = pd.read_csv(ATTR_FILE)
df.columns = df.columns.str.strip()
df['image_id'] = df['image_id'].str.strip()
print("✔ Loaded.")
print(df.head())

# Convert -1 → 0 for all attributes (easier to work with)
for col in df.columns:
    if col != "image_id":
        df[col] = df[col].replace(-1, 0)

# ============================
# APPLY HARD FILTERS
# ============================
print("\n🔍 Applying hard filters (Eyeglasses=0, Blurry=0)...")

for attr, must_be in HARDCODE_FILTERS.items():
    df = df[df[attr] == (1 if must_be == 1 else 0)]

print("Remaining after hard filters:", len(df))

# ============================
# PRIMARY CLASS SPLITTING
# ============================
print("\n📊 Splitting into primary classes...")

class_groups = {}
for class_name, attr in PRIMARY_CLASSES.items():
    class_df = df[df[attr] == 1]
    print(f"{class_name}: {len(class_df)} available")
    class_groups[class_name] = class_df

# ============================
# SOFT BALANCING FUNCTION
# ============================
def soft_balance(group_df, target_count):
    """
    Soft balance secondary attributes without strict equal splits.
    Only ensures that each subgroup >= 20% of samples.
    If fewer, upsample; if more, downsample proportionally.
    """
    MIN_RATIO = 0.20
    df_bal = group_df.copy()

    for attr in SECONDARY_SOFT_BALANCE:
        if attr not in df_bal.columns:
            continue

        # proportions
        count1 = df_bal[df_bal[attr] == 1]
        count0 = df_bal[df_bal[attr] == 0]

        total = len(df_bal)
        if total == 0:
            continue

        r1 = len(count1) / total
        r0 = len(count0) / total

        # If any subgroup too small → upsample
        if r1 < MIN_RATIO:
            need = int(MIN_RATIO * total) - len(count1)
            if need > 0:
                df_bal = pd.concat([df_bal, count1.sample(need, replace=True)])

        if r0 < MIN_RATIO:
            need = int(MIN_RATIO * total) - len(count0)
            if need > 0:
                df_bal = pd.concat([df_bal, count0.sample(need, replace=True)])

    # After soft balancing → make final strict crop to target_count
    if len(df_bal) >= target_count:
        df_bal = df_bal.sample(target_count)  
    else:
        # If not enough, allow upsampling
        df_bal = df_bal.sample(target_count, replace=True)

    return df_bal

# ============================
# APPLY SOFT BALANCING
# ============================
final_df = []

print("\n⚖ Applying SOFT balancing for each primary class...")

for class_name, group_df in class_groups.items():
    print(f"\n➡ {class_name} BEFORE soft-balance: {len(group_df)}")

    balanced_group = soft_balance(group_df, TARGET_PER_CLASS)

    print(f"➡ {class_name} AFTER soft-balance: {len(balanced_group)}")

    balanced_group["class"] = class_name
    final_df.append(balanced_group)

final_df = pd.concat(final_df)
print("\n✔ Final soft-balanced dataset size:", len(final_df))

# ============================
# SAVE IMAGES
# ============================
print("\n💾 Saving images...")
for cls in PRIMARY_CLASSES.keys():
    os.makedirs(os.path.join(OUT_DIR, cls), exist_ok=True)

count_saved = 0

for idx, row in final_df.iterrows():
    src = os.path.join(IMG_DIR, row["image_id"])
    dst = os.path.join(OUT_DIR, row["class"], row["image_id"])

    try:
        img = Image.open(src)
        img.save(dst)
        count_saved += 1
    except:
        pass

print(f"✔ Saved {count_saved} images into:")
print(OUT_DIR)
