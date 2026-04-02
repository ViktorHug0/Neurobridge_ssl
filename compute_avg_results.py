import os
import argparse

import pandas as pd

# Get input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', default="", type=str)
parser.add_argument('--output_name', default="avg_results.csv", type=str)

args = parser.parse_args()

if not os.path.exists(args.result_dir):
    raise Exception("wrong dir")

df_list = []
for run in sorted(os.listdir(args.result_dir)):
    run_path = os.path.join(args.result_dir, run)
    if not os.path.isdir(run_path):
        continue
    file = os.path.join(run_path, "result.csv")
    if not os.path.isfile(file):
        # e.g. dataset_configs/ or other non-run folders under the batch root
        continue
    df = pd.read_csv(file)
    df['sub'] = run[-6:]
    cols = ['sub'] + [col for col in df.columns if col != 'sub']
    df = df[cols]
    df_list.append(df)

if not df_list:
    raise FileNotFoundError(
        f"No result.csv files found under '{args.result_dir}'. "
        "Expected one subdirectory per fold (e.g. YYYYMMDD-HHMMSS-sub-NN/) "
        "each containing result.csv, as produced by train.py --output_dir."
    )

# Concatenate all DataFrames
all_data = pd.concat(df_list, ignore_index=True)

# Extract numeric columns (excluding 'sub', 'best epoch', and 'architecture')
numeric_cols = all_data.select_dtypes(include=['number']).columns.tolist()
# Also include columns that look like numbers but might be objects/strings
for col in all_data.columns:
    if col in ['sub', 'best epoch', 'architecture']:
        continue
    if col not in numeric_cols:
        try:
            pd.to_numeric(all_data[col])
            numeric_cols.append(col)
        except (ValueError, TypeError):
            continue

# Convert to float, keep two decimal places and pad with zeros (convert to string)
for col in numeric_cols:
    all_data[col] = all_data[col].astype(float).map(lambda x: f"{x:.1f}")

# Calculate average values (still using float for calculation, then formatting)
avg_values = all_data[numeric_cols].astype(float).mean()
avg_row = {col: f"{avg_values[col]:.1f}" for col in numeric_cols}
avg_row['sub'] = 'Average'

# Add average row
all_data = pd.concat([all_data, pd.DataFrame([avg_row])], ignore_index=True)

# Sort by sub (numeric), keep 'Average' at bottom
def extract_sub_num(x):
    if x == "Average":
        return float('inf')
    return int(''.join(filter(str.isdigit, x)))  # 提取数字

all_data = all_data.sort_values(by="sub", key=lambda col: col.map(extract_sub_num))

# Save the merged result
all_data.to_csv(os.path.join(args.result_dir, args.output_name), index=False)

print(all_data)