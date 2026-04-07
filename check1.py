import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("final_forensic_manifest.csv")

# Attribute columns
attributes = ["Arched_Eyebrows", "Bushy_Eyebrows", "Narrow_Eyes", "Bags_Under_Eyes","Male","Young"]

# --- 1. Count Distribution ---
print("COUNT DISTRIBUTION\n")

for attr in attributes:
    print(f"{attr}:")
    print(df[attr].value_counts())
    print()
