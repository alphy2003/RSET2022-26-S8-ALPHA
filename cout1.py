import pandas as pd

df = pd.read_csv(r"D:\major_phase2\final_forensic_manifest.csv")

# 1. Check for perfect overlaps (The most common error)
male_young_overlap = len(df[(df['Male'] == 1) & (df['Young'] == 1)])
female_young_overlap = len(df[(df['Male'] == 0) & (df['Young'] == 1)])

print(f"Total Images: {len(df)}")
print(f"Young Males: {male_young_overlap}")
print(f"Young Females: {female_young_overlap}")

# 2. Check the rare attributes across genders
bushy_male = len(df[(df['Bushy_Eyebrows'] == 1) & (df['Male'] == 1)])
bushy_female = len(df[(df['Bushy_Eyebrows'] == 1) & (df['Male'] == 0)])

print(f"Bushy Brows (Male): {bushy_male}")
print(f"Bushy Brows (Female): {bushy_female}")