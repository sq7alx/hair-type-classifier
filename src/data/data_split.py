import os
import sys
import csv
import random
import logging
import pandas as pd
from pathlib import Path

# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

from config.config_loader import CONFIG
from config.logging_config import get_logger, setup_logger

setup_logger(
    name="hair_type_classifier",
    level=logging.INFO,
    log_file="logs/data_split.log",
    console=True,
    file=True
)
logger = get_logger("hair_type_classifier")

input_csv = os.path.join(CONFIG['dataset']['cleaned_output_dir'], CONFIG['dataset']['cleaned_output_csv'])
output_csv = CONFIG['dataset']['split_output_csv']
splits = CONFIG['dataset']['splits']

random.seed(1984)

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

total_splits = sum(splits.values())
if abs(total_splits - 1.0) > 1e-6:
    logger.warning(f"[SPLIT] The total splits sum to {total_splits:.2f} (expected 1.0). Normalizing automatically.")
    splits = {k: v/total_splits for k, v in splits.items()}
    
try:
    df_cleaned = pd.read_csv(input_csv)
    if df_cleaned.empty:
        logger.error(f"[SPLIT] Cleaned CSV file is empty: {input_csv}")
        sys.exit(1)
except FileNotFoundError:
    logger.error(f"[SPLIT] Cleaned CSV file not found: {input_csv}")
    sys.exit(1)
    
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
df_split = pd.DataFrame(rows)
df_split.to_csv(output_csv, index=False)

logger.info(f"[SPLIT] Dataset split saved to {output_csv}")
logger.info(f"[SPLIT] Total images: {len(rows)}")

split_counts = {k: 0 for k in splits.keys()}
for row in rows:
    split_counts[row["split"]] += 1

logger.info("[SPLIT] Split statistics:")
total_rows = len(rows)
for split, count in split_counts.items():
    percent = (count / total_rows) * 100
    logger.info(f"[SPLIT] {split}: {count} ({percent:.2f}%)")
