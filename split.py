import pandas as pd
from sklearn.model_selection import train_test_split

# =======================
# 1. CONFIGURATION
# =======================
MANIFEST_PATH = r"D:\major_phase2\eye_region_manifest.csv"
OUTPUT_DIR = r"D:\major_phase2\\"

# =======================
# 2. LOAD & PREPARE
# =======================
df = pd.read_csv(MANIFEST_PATH)

# Create a combined key for perfect stratification
# We include Gender, Age, and Narrow Eyes to ensure these critical features 
# are distributed equally across Train, Val, and Test sets.
df['strat_key'] = (
    df['Male'].astype(str) + "_" + 
    df['Young'].astype(str) + "_" + 
    df['Narrow_Eyes'].astype(str)
)

# =======================
# 3. STRATIFIED SPLIT
# =======================
print(f"Splitting {len(df)} images...")

# Split 1: 80% Training, 20% Temporary (for Val/Test)
df_train, df_temp = train_test_split(
    df, 
    test_size=0.20, 
    random_state=42, 
    stratify=df['strat_key']
)

# Split 2: Divide the 20% into half Validation (10%) and half Test (10%)
df_val, df_test = train_test_split(
    df_temp, 
    test_size=0.50, 
    random_state=42, 
    stratify=df_temp['strat_key']
)

# =======================
# 4. SAVE & VERIFY
# =======================

# Remove the helper column before saving
for split in [df_train, df_val, df_test]:
    split.drop(columns=['strat_key'], inplace=True)

# Save files
df_train.to_csv(OUTPUT_DIR + "train.csv", index=False)
df_val.to_csv(OUTPUT_DIR + "val.csv", index=False)
df_test.to_csv(OUTPUT_DIR + "test.csv", index=False)

print("\n✅ Dataset Split Successfully!")
print(f"  Training Set:   {len(df_train)} images")
print(f"  Validation Set: {len(df_val)} images")
print(f"  Testing Set:    {len(df_test)} images")

# Final Balance Check
print("\nFeature Distribution (Training Set):")
for col in ['Male', 'Narrow_Eyes', 'Arched_Eyebrows']:
    perc = (df_train[col].sum() / len(df_train)) * 100
    print(f" - {col:15}: {perc:.2f}%")