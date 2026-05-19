import pandas as pd
import os

# Read the second sheet
df = pd.read_excel("dataset.xlsx", sheet_name="Raw Data")

# Remove accidental spaces in column names
df.columns = df.columns.str.strip()

# Create output folder
output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

# Split by essay_num
for essay_num, group in df.groupby("essay_num"):
    
    group.to_csv(
        os.path.join(output_dir, f"essay_{essay_num}.csv"),
        index=False
    )

print(f"Created {df['essay_num'].nunique()} CSV files")