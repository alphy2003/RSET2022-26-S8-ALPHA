import pandas as pd

# Load CSV
df = pd.read_csv("final_combined_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert -1 to 0 for specific columns
df["Big_Nose"] = df["Big_Nose"].replace(-1, 0)
df["Pointy_Nose"] = df["Pointy_Nose"].replace(-1, 0)

# Save updated file
df.to_csv("updated_nose_attributes.csv", index=False)

print("✅ Conversion completed.")
