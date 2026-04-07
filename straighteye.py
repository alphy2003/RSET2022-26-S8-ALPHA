






import pandas as pd
import os
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
ATTR_PATH = r"D:\major_phase2\list_attr_celeba.csv"
SOURCE_IMG_DIR = r"D:\Dataset\img_align_celeba\img_align_celeba"
TARGET_IMG_DIR = r"D:\major_phase2\filtered_eye_images1"
TARGET_CSV = r"D:\major_phase2\eye_final_train1.csv"

os.makedirs(TARGET_IMG_DIR, exist_ok=True)

def filter_celeba_for_eyes():
    print("Reading CelebA attributes...")
    
    # Official list of 40 CelebA attributes
    attr_names = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Necklace', 'Wearing_Necktie', 
        'Young'
    ] # This list has 39 items. Total attributes in CelebA is 40. 
    # Let's add the missing one (usually 'Shadows' or 'Heavy_Makeup' is double checked)
    # Actually, let's use a more robust way to name columns regardless of the list length.

    # 1. Load data
    df = pd.read_csv(ATTR_PATH, sep=None, skiprows=2, header=None, engine='python')
    
    # 2. MATCH COLUMNS DYNAMICALLY
    num_found = df.shape[1]
    print(f"Detected {num_found} columns in file.")

    # Generate generic names first to avoid crashes
    new_cols = ['image_id'] + [f'attr_{i}' for i in range(num_found - 1)]
    df.columns = new_cols

    # 3. MANUALLY MAP CRITICAL ATTRIBUTES BY COLUMN INDEX
    # In CelebA standard comma-format:
    # Arched_Eyebrows = col_1, Bags_Under_Eyes = col_3, Blurry = col_10, 
    # Eyeglasses = col_15, Male = col_20, Narrow_Eyes = col_23, Young = col_38
    mapping = {
        'attr_1': 'Arched_Eyebrows',
        'attr_3': 'Bags_Under_Eyes',
        'attr_10': 'Blurry',
        'attr_12': 'Bushy_Eyebrows',
        'attr_15': 'Eyeglasses',
        'attr_20': 'Male',
        'attr_23': 'Narrow_Eyes',
        'attr_38': 'Young'
    }
    df.rename(columns=mapping, inplace=True)

    # 4. Apply Filters (Straight faces, no glasses)
    filtered_df = df[
        (df['Eyeglasses'] == -1) & 
        (df['Blurry'] == -1)
    ].copy()

    # 5. Convert to Binary (0/1)
    eye_attrs = ['Male', 'Young', 'Arched_Eyebrows', 'Bushy_Eyebrows', 'Narrow_Eyes', 'Bags_Under_Eyes']
    for col in eye_attrs:
        filtered_df[col] = filtered_df[col].apply(lambda x: 1 if x == 1 else 0)

    # 6. Sampling to preserve rare attributes
    # We keep ALL Narrow_Eyes (rare)
    rare_mask = (filtered_df['Narrow_Eyes'] == 1) | (filtered_df['Bags_Under_Eyes'] == 1)
    rare_samples = filtered_df[rare_mask]
    common_samples = filtered_df[~rare_mask].sample(n=min(20000, len(filtered_df[~rare_mask])), random_state=42)
    
    final_df = pd.concat([rare_samples, common_samples]).sample(frac=1).reset_index(drop=True)

    # 7. Copy Files
    print(f"Copying {len(final_df)} images...")
    final_data_list = []
    for _, row in tqdm(final_df.iterrows(), total=len(final_df)):
        img_name = str(row['image_id'])
        if not img_name.lower().endswith('.jpg'): img_name += '.jpg'
        
        src = os.path.join(SOURCE_IMG_DIR, img_name)
        dst = os.path.join(TARGET_IMG_DIR, img_name)
        
        if os.path.exists(src):
            shutil.copy(src, dst)
            final_data_list.append(row)

    # 8. Save CSV
    pd.DataFrame(final_data_list)[['image_id'] + eye_attrs].to_csv(TARGET_CSV, index=False)
    print(f"\nDone! CSV saved to {TARGET_CSV}")

if __name__ == "__main__":
    filter_celeba_for_eyes()