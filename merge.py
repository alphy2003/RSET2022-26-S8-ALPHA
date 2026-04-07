import pandas as pd

# Load both CSV files
csv1 = pd.read_csv("frontal_images.csv")              # Only image_id column
csv2 = pd.read_csv("eye_derived_attributes.csv")        # Eye attributes file

# Make sure column name is exactly same
print(csv1.columns)
print(csv2.columns)

# Intersection using inner merge
intersection_df = pd.merge(csv1, csv2, on="image_id", how="inner")

# Save result
intersection_df.to_csv("intersection_output.csv", index=False)

print("Intersection CSV created successfully!")
print("Total images after intersection:", len(intersection_df))