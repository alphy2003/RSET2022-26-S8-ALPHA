import pandas as pd

# Load your CSV file
df = pd.read_csv("merged.csv")

# Remove images where Wearing_Eyeglasses == 1
df_no_glasses = df[df["Eyeglasses"] == -1]

# Save the new filtered CSV
df_no_glasses.to_csv("filtered_no_eyeglasses.csv", index=False)

print("Original dataset size:", len(df))
print("After removing eyeglasses:", len(df_no_glasses))
