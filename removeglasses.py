import pandas as pd

# Load CelebA attribute file
celeba_df = pd.read_csv("list_attr_celeba.csv")

# Load your eye dataset
eye_df = pd.read_csv("intersection_output.csv")

# Clean column names (important)
celeba_df.columns = celeba_df.columns.str.strip()
eye_df.columns = eye_df.columns.str.strip()

# CelebA uses -1 and 1
# 1 = Has attribute
# -1 = Does not have attribute

# Keep only images WITHOUT eyeglasses
celeba_no_glasses = celeba_df[celeba_df["Eyeglasses"] == -1]

# Now intersect with your eye dataset
filtered_eye_df = eye_df[eye_df["image_id"].isin(celeba_no_glasses["image_id"])]

# Save cleaned dataset
filtered_eye_df.to_csv("final_eye_attributes_no_glasses.csv", index=False)

print("Glasses removed successfully!")
print("Remaining images:", len(filtered_eye_df))