import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
CSV_PATH = r"D:\major_phase2\train.csv"  # Path to your CSV
OUTPUT_DIR = r"D:\major_phase2\visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_visualizations():
    # 1. Load Data
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {CSV_PATH}")
        return

    # 2. Filter attribute columns (removing ID/Filename)
    # This keeps only binary features like Big_Eyes, Male, etc.
    cols_to_exclude = ['image_id', 'filename', 'Unnamed: 0']
    attr_cols = [c for c in df.columns if c not in cols_to_exclude]
    
    # --- PART A: CATEGORY DISTRIBUTION GRAPH ---
    counts = df[attr_cols].sum().sort_values(ascending=False).reset_index()
    counts.columns = ['Attribute', 'Frequency']
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create bar plot
    ax = sns.barplot(data=counts, x='Attribute', y='Frequency', palette='viridis')
    
    # Add labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontweight='bold')

    plt.title('Eye Dataset: Attribute Distribution', fontsize=16, pad=15)
    plt.xlabel('Forensic Category', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    dist_path = os.path.join(OUTPUT_DIR, 'eye_distribution_graph.png')
    plt.savefig(dist_path, dpi=300)
    print(f"Distribution graph saved to: {dist_path}")

    # --- PART B: CORRELATION MATRIX HEATMAP ---
    # Correlation (r) ranges from -1 to 1
    corr_matrix = df[attr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    # 'coolwarm' is best for correlation to show positive vs negative relationships
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".2f", 
                linewidths=0.5, 
                vmin=-1, vmax=1, 
                center=0)
    
    plt.title('Eye Attribute Correlation Matrix', fontsize=16, pad=15)
    plt.tight_layout()
    
    corr_path = os.path.join(OUTPUT_DIR, 'eye_correlation_matrix.png')
    plt.savefig(corr_path, dpi=300)
    print(f"Correlation matrix saved to: {corr_path}")

if __name__ == "__main__":
    generate_visualizations()