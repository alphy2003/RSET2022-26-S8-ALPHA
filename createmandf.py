import pandas as pd

# 1. Use the path to your Excel file
# The 'r' before the path helps Python handle the backslashes correctly
file_path = r'C:\Users\User\Downloads\IV LIST.xlsx'

# 2. Use pd.read_excel instead of pd.read_csv
# dtype=str keeps your phone numbers and long IDs exactly as they are
df = pd.read_excel(file_path, dtype=str)

# 3. Clean column names
df.columns = df.columns.str.strip()

# Print columns so you can verify the name of your gender column
print("Columns in your file:", df.columns.tolist())

# 4. Identify the gender column (Change 'Gender' if your column is named differently)
gender_col = 'GENDER' 

if gender_col not in df.columns:
    print(f"Error: Could not find '{gender_col}'. Please check the column names printed above.")
else:
    # 5. Filter the data (M for Male, F for Female)
    # This handles "Male", "M", "female", etc.
    male_df = df[df[gender_col].fillna('').str.strip().str.upper().str.startswith('M')]
    female_df = df[df[gender_col].fillna('').str.strip().str.upper().str.startswith('F')]

    # 6. Save to new Excel files
    male_df.to_excel('male_records1.xlsx', index=False)
    female_df.to_excel('female_records1.xlsx', index=False)

    print(f"Success! Created 'male_records.xlsx' ({len(male_df)} rows) and 'female_records.xlsx' ({len(female_df)} rows).")