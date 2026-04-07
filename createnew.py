import pandas as pd

# 1. Load your current 46k attribute file
# Replace with your actual path
CSV_PATH = r"D:\major_phase2\train.csv" 
df = pd.read_csv(CSV_PATH)

# Ensure attributes are 1 and 0 (CelebA often uses 1 and -1)
cols_to_fix = ['Male', 'Young', 'Arched_Eyebrows', 'Bushy_Eyebrows', 'Narrow_Eyes', 'Bags_Under_Eyes']
for col in cols_to_fix:
    df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)

# 2. Define the Target Size
TARGET_SIZE = 12000
per_bucket = TARGET_SIZE // 4 # 3,000 images per bucket

# 3. Create 4 Buckets for Core Balance
m_young = df[(df['Male'] == 1) & (df['Young'] == 1)]
m_old = df[(df['Male'] == 1) & (df['Young'] == 0)]
f_young = df[(df['Male'] == 0) & (df['Young'] == 1)]
f_old = df[(df['Male'] == 0) & (df['Young'] == 0)]

# 4. Sample 3,000 from each (or as many as available)
balanced_12k = pd.concat([
    m_young.sample(n=min(len(m_young), per_bucket), random_state=42),
    m_old.sample(n=min(len(m_old), per_bucket), random_state=42),
    f_young.sample(n=min(len(f_young), per_bucket), random_state=42),
    f_old.sample(n=min(len(f_old), per_bucket), random_state=42)
])

# 5. Shuffle and Save
balanced_12k = balanced_12k.sample(frac=1).reset_index(drop=True)
balanced_12k.to_csv(r"D:\major_phase2\balanced_forensic_12k.csv", index=False)

# 6. Print the result
print("--- 12k Balanced Distribution ---")
for col in cols_to_fix:
    pos = balanced_12k[col].sum()
    print(f"{col:15}: {pos} images ({(pos/len(balanced_12k))*100:.1f}%)")