import pandas as pd

# =======================
# 1. CONFIGURATION
# =======================
FILTERED_IMAGES_CSV = r"D:\major_phase2\final_soft_balanced.csv"
CELEBA_ATTR_FILE = r"D:\major_phase2\list_attr_celeba.csv"
OUTPUT_CSV = r"D:\major_phase2\eye_region_manifest.csv"

# The standard 40 CelebA attributes in their exact order
CELEBA_ATTRIBUTES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 
    'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

EYE_ATTRIBUTES = ['Arched_Eyebrows', 'Bushy_Eyebrows', 'Narrow_Eyes', 'Bags_Under_Eyes', 'Male', 'Young']

# =======================
# 2. LOAD DATA
# =======================
print("Loading files...")

df_filtered = pd.read_csv(FILTERED_IMAGES_CSV)

# We read the file, skipping the first TWO lines (count and likely broken header)
# Then we manually assign the columns.
try:
    # We skip 2 lines to get straight to the data
    df_all_attr = pd.read_csv(CELEBA_ATTR_FILE, skiprows=2, header=None, sep=None, engine='python')
    
    # Assign our manual header list (image_id + 40 attributes)
    df_all_attr.columns = ['image_id'] + CELEBA_ATTRIBUTES
    print(f"✅ Successfully mapped {len(df_all_attr.columns)} columns.")

except Exception as e:
    print(f"❌ Error during manual mapping: {e}")
    # Fallback: if skip 2 is too many, try skip 1
    df_all_attr = pd.read_csv(CELEBA_ATTR_FILE, skiprows=1, header=None, sep=None, engine='python')
    df_all_attr.columns = ['image_id'] + CELEBA_ATTRIBUTES

# =======================
# 3. MERGE & CLEAN
# =======================
print("Combining attributes with your image list...")

# Inner merge to keep only your filtered images
final_df = pd.merge(
    df_filtered[['image_id']], 
    df_all_attr[['image_id'] + EYE_ATTRIBUTES], 
    on='image_id', 
    how='inner'
)

# Convert values to 0 and 1
for col in EYE_ATTRIBUTES:
    final_df[col] = pd.to_numeric(final_df[col]).replace(-1, 0)

# =======================
# 4. SAVE & VERIFY
# =======================
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ SUCCESS!")
print(f"Total Images in Manifest: {len(final_df)}")
print("-" * 30)
print(final_df.head())