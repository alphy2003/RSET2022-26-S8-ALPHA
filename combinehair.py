import pandas as pd

def merge_hair_datasets(csv_path_1, csv_path_2, output_path):
    # 1. Load both CSV files
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    print(f"CSV 1 Rows: {len(df1)}")
    print(f"CSV 2 Rows: {len(df2)}")

    # 2. Merge based on 'image_id'
    # 'how=inner' ensures we only keep images present in both files
    combined_df = pd.merge(df1, df2, on='image_id', how='inner')

    # 3. Remove any duplicate columns if they exist (except image_id)
    # This cleans up the file if both CSVs had the same attribute columns
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    # 4. Save the combined master list
    combined_df.to_csv(output_path, index=False)

    print(f"Combined Dataset Size: {len(combined_df)}")
    print(f"Combined CSV saved to: {output_path}")
    
    # Show the first few rows to verify
    print("\nPreview of combined data:")
    print(combined_df.head())

if __name__ == "__main__":
    # Update these paths to your actual file locations
    FILE_A = r"D:\major_phase2\front_facing_hair_dataset.csv"
    FILE_B = r"D:\major_phase2\hair_attributes2.csv"
    RESULT_FILE = r"D:\major_phase2\final_training_metadata.csv"

    merge_hair_datasets(FILE_A, FILE_B, RESULT_FILE)