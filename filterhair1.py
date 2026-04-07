import pandas as pd

# Load original dataset
df_original = pd.read_csv("clean_hair_dataset.csv")

# Load generated attributes
df_new = pd.read_csv("hair_attributes1.csv")

# Keep only needed columns
df_length = df_new[["image_id", "hair_length"]]

# Merge on image_id
df_merged = df_original.merge(
    df_length,
    on="image_id",
    how="left"  # keeps all original rows
)

# Save new dataset
df_merged.to_csv("final_hair_dataset1.csv", index=False)

print("✅ Hair length merged successfully!")
