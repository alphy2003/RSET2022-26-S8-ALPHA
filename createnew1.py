import pandas as pd

df = pd.read_csv(r"D:\major_phase2\list_attr_celeba.csv")

# Cleanup: Convert -1 to 0
cols = ['Eyeglasses', 'Male', 'Young', 'Arched_Eyebrows', 'Bushy_Eyebrows', 'Narrow_Eyes', 'Bags_Under_Eyes']
for col in cols:
    df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)

def get_balanced_subset(data, target_total):
    per_bucket = target_total // 4
    
    # Define buckets
    m_y = data[(data['Male'] == 1) & (data['Young'] == 1)]
    m_o = data[(data['Male'] == 1) & (data['Young'] == 0)]
    f_y = data[(data['Male'] == 0) & (data['Young'] == 1)]
    f_o = data[(data['Male'] == 0) & (data['Young'] == 0)]
    
    buckets = [m_y, m_o, f_y, f_o]
    sampled_buckets = []
    
    for b in buckets:
        # If bucket is smaller than 1500, take all of it. Otherwise, take 1500.
        count_to_take = min(len(b), per_bucket)
        sampled_buckets.append(b.sample(n=count_to_take, random_state=42))
        
    return pd.concat(sampled_buckets)

# 1. Get Spects (likely limited by Young categories)
spects_pool = df[df['Eyeglasses'] == 1]
balanced_spects = get_balanced_subset(spects_pool, 6000)

# 2. Get Non-Spects (plenty of data here, will hit exactly 6000)
no_spects_pool = df[df['Eyeglasses'] == 0]
balanced_no_spects = get_balanced_subset(no_spects_pool, 6000)

# 3. Combine
final_df = pd.concat([balanced_spects, balanced_no_spects]).sample(frac=1).reset_index(drop=True)
final_df.to_csv(r"D:\major_phase2\ultra_balanced_final.csv", index=False)

print(f"Dataset created with {len(final_df)} images.")
print("--- Internal Distribution of Spects Category ---")
print(balanced_spects[['Male', 'Young']].value_counts())