import pandas as pd

# Load CSV
df = pd.read_csv("balanced_hair_dataset.csv")

# Remove image_id column
attribute_columns = df.columns[1:]

# Count distribution (number of 1s in each column)
attribute_counts = df[attribute_columns].sum()

print("Attribute Distribution:\n")
print(attribute_counts)

# Optional: percentage distribution
print("\nPercentage Distribution:\n")
print((attribute_counts / len(df)) * 100)