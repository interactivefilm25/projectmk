import os
import pandas as pd

csv_dir = 'data/csv'
output_file = 'data/combined_features.csv'

all_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
dfs = [pd.read_csv(os.path.join(csv_dir, f)) for f in all_files]

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv(output_file, index=False)

print(f"Combined {len(dfs)} files into {output_file}")
