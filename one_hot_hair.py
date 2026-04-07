import pandas as pd

# Load CSV
df = pd.read_csv(r"D:\major_phase2\hair_gan_attributes.csv")

# One-hot encode with integer values
df_encoded = pd.get_dummies(
    df,
    columns=["length", "volume"],
    prefix=["length", "volume"],
    dtype=int   # 👈 forces 0/1 instead of True/False
)

# Save
df_encoded.to_csv("hair_dataset_onehot.csv", index=False)

print("✅ One-hot encoding done with 0/1 values")
print(df_encoded.head())
