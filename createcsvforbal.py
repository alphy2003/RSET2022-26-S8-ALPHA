import pandas as pd
import random

# ---- Paths ----
ATTR_FILE = r"D:\major_phase2\list_attr_celeba.csv"
OUTPUT_CSV = "final_soft_balanced.csv"

# ---- Load attributes ----
df = pd.read_csv(ATTR_FILE)

# Convert -1 to 0 (optional but useful)
for col in df.columns[1:]:
    df[col] = df[col].replace(-1, 0)
df = df[df["Eyeglasses"] == 0]
# Attributes you selected
attrs = [
    "Male",
    "Young",
    "Narrow_Eyes",
    "Smiling",
    "Bushy_Eyebrows",
    "Arched_Eyebrows",
    "Heavy_Makeup"
]

print("\nSelected attributes:", attrs)

# ---- Count positives for each attribute ----
counts = {a: df[a].sum() for a in attrs}
min_count = min(counts.values())   # soft balance target

print("\n=== Attribute sample counts before balancing ===")
for a, c in counts.items():
    print(f"{a:17}: {c}")

print(f"\nSoft-balance target per attribute: {min_count}")

# ---- Collect balanced samples for each attribute ----
selected = set()

for a in attrs:
    positives = df[df[a] == 1]["image_id"].tolist()
    random.shuffle(positives)
    chosen = positives[:min_count]
    selected.update(chosen)

print(f"\nTotal unique images selected after soft balancing: {len(selected)}")

# ---- Save final list ----
final_df = pd.DataFrame({"image_id": sorted(selected)})
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n📄 Saved soft-balanced list to: {OUTPUT_CSV}")
