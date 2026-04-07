import pandas as pd

# Load your eye dataset (already filtered no glasses)
eye_df = pd.read_csv("eye_attributes_onehot.csv")

# Load original CelebA attribute file
celeba_df = pd.read_csv("list_attr_celeba.csv")

# Clean column names
eye_df.columns = eye_df.columns.str.strip()
celeba_df.columns = celeba_df.columns.str.strip()

# Select only required attributes
selected_attrs = celeba_df[[
    "image_id",
    "Male",
    "Young",
    "Narrow_Eyes",
    "Bags_Under_Eyes"
]]

# Merge based on image_id
merged_df = pd.merge(
    eye_df,
    selected_attrs,
    on="image_id",
    how="inner"
)

# Convert -1 to 0 (important!)
for col in ["Male", "Young", "Narrow_Eyes", "Bags_Under_Eyes"]:
    merged_df[col] = merged_df[col].replace(-1, 0)

# Save updated dataset
merged_df.to_csv("final_eye_attributes_with_celeba.csv", index=False)

print("Attributes merged successfully!")
print("Total rows:", len(merged_df))
print(merged_df.head())