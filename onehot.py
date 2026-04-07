import pandas as pd

# Load your dataset
df = pd.read_csv("final_eye_attributes_no_glasses.csv")

# One-hot encode
one_hot_df = pd.get_dummies(
    df,
    columns=["eye_shape", "orientation", "hooded"],
    dtype=int   # 👈 This forces 0 and 1
)

# Save
one_hot_df.to_csv("eye_attributes_onehot.csv", index=False)

print("One-hot encoding done!")
print(one_hot_df.head())