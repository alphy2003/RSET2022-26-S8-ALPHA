import pandas as pd

# -----------------------------
# Hair-specific attributes
# -----------------------------
hair_specific_labels = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
    'Gray_Hair', 'Male', 'Receding_Hairline',
    'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat'
]

# -----------------------------
# Function to create hair CSV
# -----------------------------
def create_hair_csv(input_csv, output_csv="hair_attributes.csv"):
    
    # Load original CSV
    df = pd.read_csv(input_csv)

    # Safety check
    required_cols = ['image_id'] + hair_specific_labels
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Filter columns
    hair_df = df[required_cols].copy()

    # Convert (-1,1) → (0,1)
    hair_df[hair_specific_labels] = hair_df[hair_specific_labels].replace(-1, 0)

    # Save new CSV
    hair_df.to_csv(output_csv, index=False)

    print(f"✅ Hair attribute CSV created: {output_csv}")
    print(f"✅ Total samples: {len(hair_df)}")
    print(f"✅ Total attributes: {len(hair_specific_labels)}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    create_hair_csv(r"D:\major_phase2\list_attr_celeba.csv", "hair_attributes.csv")
