import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\major_phase2\nose_full_balanced.csv")

# Remove Pale_Skin column
df = df.drop(columns=["Pale_Skin"])

# (Optional) save updated dataset
df.to_csv("nose_dataset_cleaned.csv", index=False)

print("Pale_Skin column removed successfully.")
print(df.head())
