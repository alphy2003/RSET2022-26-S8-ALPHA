import pandas as pd

INPUT = r"D:\major_phase2\hair_dataset_onehot.csv"
OUTPUT =r"D:\major_phase2\balanced_hair_dataset.csv"


TARGET = 8000  # per class

df = pd.read_csv(INPUT)

###################################
# HELPER FUNCTION
###################################
def sample_class(col):
    subset = df[df[col] == 1]
    return subset.sample(
        min(TARGET, len(subset)),
        random_state=42
    )

###################################
# SAMPLE EACH ATTRIBUTE
###################################

parts = []

cols = [
    "length_short","length_medium","length_long",
    "volume_thin","volume_normal","volume_thick",
    "Straight_Hair","Wavy_Hair"
]

for c in cols:
    parts.append(sample_class(c))

###################################
# MERGE & REMOVE DUPLICATES
###################################

balanced = pd.concat(parts).drop_duplicates("image_id")

###################################
# MILD BALD CONTROL
###################################

bald_yes = balanced[balanced["Bald"]==1]
bald_no  = balanced[balanced["Bald"]==0]

if len(bald_yes) > len(bald_no)//4:
    bald_yes = bald_yes.sample(len(bald_no)//4, random_state=42)

balanced = pd.concat([bald_yes,bald_no])

###################################
# FINAL SHUFFLE
###################################

balanced = balanced.sample(frac=1, random_state=42)

balanced.to_csv(OUTPUT,index=False)

print("✅ Balanced CSV created:", len(balanced))

###################################
# REPORT
###################################

print("\nDistribution:")
for c in ["Bald"] + cols:
    print(f"{c}: {balanced[c].mean()*100:.2f}%")
