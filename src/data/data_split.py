import os
import sys
import csv
import random
import yaml
import pandas as pd
from pathlib import Path

# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

from config.config_loader import CONFIG

input_csv = os.path.join(CONFIG['dataset']['cleaned_output_dir'], CONFIG['dataset']['cleaned_output_csv'])
print(input_csv)
output_csv = CONFIG['dataset']['split_output_csv']
splits = CONFIG['dataset']['splits']

random.seed(1984)

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

total_splits = sum(splits.values())

if total_splits != 1.0:
    print(f"Warning! The total splits value is {total_splits:.2f}")
    choice = input("Do you want to continue? (y/n)").lower()
    if choice != "y":
        print("Exited")
        exit(1)
try:
    df_cleaned = pd.read_csv(input_csv)
    if df_cleaned.empty:
        print(f"Error: Cleaned CSV file is empty: {input_csv}")
        exit(1)
except FileNotFoundError:
    print(f"Error: Cleaned CSV file not found: {input_csv}")
    exit(1)
    
rows = []
total_rows = len(df_cleaned)

grouped = df_cleaned.groupby(['class', 'subclass'])

for (class_name, subclass_name), group in grouped:  # 1/2/3
    indices = group.index.tolist()
    random.shuffle(indices)
    
    start = 0
    total = len(indices)
    
    for i, (split, ratio) in enumerate(splits.items()):
        if i == len(splits) - 1:
            end = total
        else:
            end = start + int(ratio * total)
        
        for idx in indices[start:end]:
            row = df_cleaned.loc[idx]
            rows.append({
                "class": row["class"],
                "subclass": row["subclass"],
                "filename": row["filename"],
                "split": split,
                "path": row["path"]
            })
        start = end         
        

# Save CSV
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["class", "subclass", "filename", "split", "path"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Dataset split saved to {output_csv}")
print(f"Total images: {len(rows)}")

split_counts = {"train": 0, "val": 0, "test": 0}
split_parts = [splits["train"], splits["val"], splits["test"]]

for row in rows:
    split_counts[row["split"]] += 1

print("Split statistics:")
for split, count in split_counts.items():
    percent = (count / len(rows)) * 100
    print(f"{split}: {count} ({percent:.2f}%)")
