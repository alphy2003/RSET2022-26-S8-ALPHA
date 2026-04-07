import pandas as pd

# Load your CSV
df = pd.read_csv("categorized_faces.csv")

# One-hot encode face_shape column
one_hot = pd.get_dummies(df['face_shape']).astype(int)

# Combine with image_id
df_encoded = pd.concat([df['image_id'], one_hot], axis=1)

# Save new CSV
df_encoded.to_csv("face_shape_onehot.csv", index=False)

print("One-hot encoded CSV with 0/1 created!")