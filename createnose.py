import pandas as pd
import os

# --- CONFIGURATION ---
ATTR_PATH = r"D:\major_phase2\list_attr_celeba.csv" 
OUTPUT_CSV = r"D:\major_phase2\nose_full_balanced.csv"

# Target attributes for the nose
TARGET_ATTRS = ['Big_Nose', 'Pointy_Nose', 'Male', 'Young', 'Pale_Skin']

def balance_full_dataset():
    print("Loading 202,599 attributes...")
    df = pd.read_csv(ATTR_PATH)
    
    # 1. Convert -1 to 0
    for attr in TARGET_ATTRS:
        df[attr] = df[attr].replace(-1, 0)
    
    # 2. Group by the most distinctive features
    # We use Big_Nose, Pointy_Nose, and Male to create 8 unique 'types' of noses
    groups = df.groupby(['Big_Nose', 'Pointy_Nose', 'Male'])
    
    # Find the size of the smallest group to determine how many we can take per group
    # Usually, 'Female + Big Nose' is the rarest category in CelebA
    min_size = groups.size().min()
    print(f"Smallest group size found: {min_size}")

    # 3. Sample equally from all groups
    # We will take up to 8,000 images per group to get a massive balanced set
    limit_per_group = min(min_size, 2667) 
    balanced_df = groups.apply(lambda x: x.sample(limit_per_group)).reset_index(drop=True)
    
    # 4. Save
    balanced_df = balanced_df[['image_id'] + TARGET_ATTRS]
    balanced_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Created a balanced dataset of {len(balanced_df)} images.")
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    balance_full_dataset()