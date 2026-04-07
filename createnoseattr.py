import pandas as pd

# -------------------------------------------------
# 1️⃣ Load landmarks CSV (with required columns)
# -------------------------------------------------
df = pd.read_csv(r"D:\major_phase2\archive (1)\list_landmarks_align_celeba.csv")

df.columns = df.columns.str.strip()

# -------------------------------------------------
# 2️⃣ Compute Facial Reference Points
# -------------------------------------------------

# Eye midpoint (vertical)
df["eye_mid_y"] = (df["lefteye_y"] + df["righteye_y"]) / 2

# Mouth midpoint (vertical)
df["mouth_mid_y"] = (df["leftmouth_y"] + df["rightmouth_y"]) / 2

# -------------------------------------------------
# 3️⃣ Compute Nose Length Ratio
# -------------------------------------------------

df["nose_length"] = df["nose_y"] - df["eye_mid_y"]
df["face_height"] = df["mouth_mid_y"] - df["eye_mid_y"]

df["nose_length_ratio"] = df["nose_length"] / df["face_height"]

# -------------------------------------------------
# 4️⃣ Compute Nose Width Ratio
# -------------------------------------------------

df["eye_distance"] = df["righteye_x"] - df["lefteye_x"]

df["nose_width_ratio"] = df["eye_distance"] / df["face_height"]

# -------------------------------------------------
# 5️⃣ Compute Thresholds Using Percentiles
# -------------------------------------------------

length_high = df["nose_length_ratio"].quantile(0.70)
length_low  = df["nose_length_ratio"].quantile(0.30)

width_high  = df["nose_width_ratio"].quantile(0.75)
width_low   = df["nose_width_ratio"].quantile(0.25)

# -------------------------------------------------
# 6️⃣ Create Binary Attributes (0/1)
# -------------------------------------------------

df["Long_Nose"]   = (df["nose_length_ratio"] >= length_high).astype(int)
df["Short_Nose"]  = (df["nose_length_ratio"] <= length_low).astype(int)

df["Wide_Nose"]   = (df["nose_width_ratio"] >= width_high).astype(int)
df["Narrow_Nose"] = (df["nose_width_ratio"] <= width_low).astype(int)

# -------------------------------------------------
# 7️⃣ Create Final CSV
# -------------------------------------------------

final_df = df[[
    "image_id",
    "Long_Nose",
    "Short_Nose",
    "Wide_Nose",
    "Narrow_Nose"
]]

final_df.to_csv("custom_nose_attributes.csv", index=False)

print("✅ CSV file created successfully.")
print("Total images:", len(final_df))
