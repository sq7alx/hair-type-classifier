import os
import time
import shutil
import hashlib
import numpy as np
import pandas as pd
import logging

from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

from config.config_loader import CONFIG
from config.logging_config import get_logger, setup_logger

setup_logger(
    name="hair_type_classifier",
    level=logging.INFO,
    log_file="logs/data_cleaning.log",
    console=True,
    file=True
)
logger = get_logger("hair_type_classifier")
start_time = time.time()

# Directories and Parameters
input_dir = CONFIG['dataset']['raw_input_dir']
output_dir = CONFIG['dataset']['cleaned_output_dir']
metadata_path = os.path.join(output_dir, CONFIG['dataset']['cleaned_output_csv'])
MIN_SIZE = CONFIG['dataset']['min_image_size']
MAX_WORKERS = CONFIG['dataset']['max_workers']

valid_extensions = (".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp", ".heic", ".png")

os.makedirs(output_dir, exist_ok=True)

hashes = {}  # for duplicates
hashes_lock = Lock()

stats = {
    'total': 0,
    'kept': 0,
    'too_small': 0,
    'single_color': 0,
    'duplicates': 0,
    'errors': 0
}

stats_lock = Lock()

duplicate_files = []
duplicate_files_lock = Lock()

metadata_records = []
metadata_lock = Lock()

def update_stats(key: str, value: int = 1):
    """Thread-safe update for stats"""
    with stats_lock:
        stats[key] = stats.get(key, 0) + value
        
def compute_perceptual_hash(img):
    """Fast perceptual hash for duplicate detection"""
    img = img.convert("RGB")
    small = img.resize((20, 20), Image.LANCZOS).convert("L")
    arr = np.array(small)
    return hashlib.md5(arr.tobytes()).hexdigest()

def is_single_color(img):
    """Check if an image is essentially a single color"""
    small = img.resize((50, 50), Image.LANCZOS)
    arr = np.array(small)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return np.all(arr == arr[0, 0, :])
    elif arr.ndim == 2:
        return np.all(arr == arr[0, 0])
    return False

def validate_image(path):
    """Validate an image file to determine whether it should be kept or skipped."""
    try:
        with Image.open(path) as img:
            img.verify()
            
        with Image.open(path) as img:
            img = img.convert("RGB")
            if min(img.size) < MIN_SIZE:
                update_stats('too_small')
                return "too_small", None
            if is_single_color(img):
                update_stats('single_color')
                return "single_color", None
            phash = compute_perceptual_hash(img)
            return "ok", phash
    except Exception as e:
        update_stats('errors')
        logger.warning(f"[CLEANING] Failed to validate image: {path} | {e}")
        return "error", None

def process_image(filepath, current_class_name, current_subclass_name):
    update_stats('total')
    status, phash = validate_image(filepath)
    
    if status != "ok":
        return None
    
    with hashes_lock:
        if phash in hashes:
            original = hashes[phash]
            with duplicate_files_lock:
                duplicate_files.append((filepath, original))
            update_stats('duplicates')
            return None
        hashes[phash] = filepath
    
    with Image.open(filepath) as img:
        img = img.convert("RGB")
        update_stats('kept')

        with metadata_lock:
            metadata_records.append({
                "class": current_class_name,
                "subclass": current_subclass_name,
                "filename": os.path.basename(filepath),
                "hash": phash,
                "width": img.width,
                "height": img.height,
                "path": os.path.relpath(filepath, input_dir)
            })

    return filepath

logger.info("[CLEANING] Started")
logger.info(f"[CLEANING] Input directory: {input_dir}")
logger.info(f"[CLEANING] Output directory: {output_dir}")
logger.info(f"[CLEANING] Min image size: {MIN_SIZE}")
logger.info(f"[CLEANING] Workers: {MAX_WORKERS}")

for class_name in os.listdir(input_dir): #1/2/3
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    for subclass_name in os.listdir(class_path): #1a,1b,1c etc.
        subfolder_input_path = os.path.join(class_path, subclass_name)
        if not os.path.isdir(subfolder_input_path):
            continue

        logger.info(f"[CLEANING] Cleaning {class_name}/{subclass_name}")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for filename in os.listdir(subfolder_input_path):
                if not filename.lower().endswith(valid_extensions):
                    continue
                filepath = os.path.join(subfolder_input_path, filename)
                futures.append(executor.submit(process_image, filepath, class_name, subclass_name))

            for f in as_completed(futures):
                result = f.result()
                if result:
                    logger.debug(f"[CLEANING] Kept: {result}")

# save metadata
if metadata_records:
    df = pd.DataFrame(metadata_records)
    df.to_csv(metadata_path, index=False)
    logger.info(f"[CLEANING] Metadata saved to {metadata_path}")


end_time = time.time()
elapsed_time = end_time - start_time

# Summary
logger.info("[CLEANING] Summary:")
total = stats['total'] if stats['total'] > 0 else 1
for key, value in stats.items():
    logger.info(f"[CLEANING] {key}: {value} ({value/total*100:.1f}%)")

if duplicate_files:
    logger.warning("\n[CLEANING] Duplicate files detected:")
    for dup, original in duplicate_files:
        logger.info(f"[CLEANING] - {dup} (same as {original})")
else:
    logger.info("\n[CLEANING]No duplicate files detected.")

logger.info(f"\n[CLEANING] Cleaning time: {elapsed_time:.2f} seconds")