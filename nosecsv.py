import pandas as pd

# ---------------------------------
# 1️⃣ Load both CSV files
# ---------------------------------
df1 = pd.read_csv("filtered_no_eyeglasses.csv")      # your filtered images
df2 = pd.read_csv("list_attr_celeba.csv")      # full attribute file

# ---------------------------------
# 2️⃣ Clean column names
# ---------------------------------
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# ---------------------------------
# 3️⃣ Keep only nose attributes from df2
# ---------------------------------
df2_nose = df2[["image_id", "Big_Nose", "Pointy_Nose"]]

# ---------------------------------
# 4️⃣ Merge (INNER JOIN = intersection)
# ---------------------------------
final_df = pd.merge(df1[["image_id"]], df2_nose, on="image_id", how="inner")

# ---------------------------------
# 5️⃣ Save result
# ---------------------------------
final_df.to_csv("nose_attributes_intersection.csv", index=False)

print("✅ New CSV created successfully.")
print("Total images in intersection:", len(final_df))
