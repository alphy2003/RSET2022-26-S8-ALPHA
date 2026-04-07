import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# Path to your CSV files
csv_files = glob.glob("eye_final_train1.csv")

# Load and combine all CSV files
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Select only numeric columns
numeric_df = combined_df.select_dtypes(include='number')

# Compute correlation matrix
corr_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)

plt.title("Correlation Matrix")
plt.tight_layout()

# Save as image
plt.savefig("correlation_matrix.png", dpi=300)

# Optional: show the plot
plt.show()