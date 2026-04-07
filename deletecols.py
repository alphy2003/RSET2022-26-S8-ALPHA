import pandas as pd

df = pd.read_csv("final_forensic_manifest.csv")

# Keep only images without eyeglasses
df_no_glasses = df[df["Eyeglasses"] == 0]

df_no_glasses.to_csv("filtered_attributes.csv", index=False)
