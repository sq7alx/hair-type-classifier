import os
import sys
import re
import yaml
import time
import shutil
import hashlib
import numpy as np
import pandas as pd

from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from pathlib import Path

from config.config_loader import CONFIG

start_time = time.time()

# Directories and Parameters
input_dir = CONFIG['dataset']['raw_input_dir']
output_dir = CONFIG['dataset']['cleaned_output_dir']
temp_bin_dir = CONFIG['dataset']['temp_bin_dir']
metadata_path = os.path.join(output_dir, CONFIG['dataset']['metadata_filename'])
MIN_SIZE = CONFIG['dataset']['min_image_size']
MAX_WORKERS = CONFIG['dataset']['max_workers']

valid_extensions = (".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp", ".heic", ".png")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_bin_dir, exist_ok=True)

hashes = {}  # for duplicates
hashes_lock = Lock()

stats = {
    'total': 0,
    'saved': 0,
    'too_small': 0,
    'single_color': 0,
    'duplicates': 0,
    'errors': 0,
    'replaced': 0
}

stats_lock = Lock()

duplicate_files = []
duplicate_files_lock = Lock()

metadata_records = []
metadata_lock = Lock()

# format check (if e.g. 1a-1.png)
pattern = re.compile(r'^([a-zA-Z0-9_]+)-\d+\.(jpg|jpeg|bmp|tiff|gif|webp|heic|png)$', re.IGNORECASE)

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
    """Validate an image file to determine whether it should be kept, skipped, or marked as duplicate."""
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
        print(f"Error validating {path}: {e}")
        return "error", None


def process_image(filename, subfolder_input_path, subfolder_output_path, current_class_name, current_subclass_name, counter):
    input_path = os.path.join(subfolder_input_path, filename)
    
    update_stats('total')
    status, phash = validate_image(input_path)
    
    if status != "ok":
        return None
    
    with hashes_lock:
        if phash in hashes:
            original = hashes[phash]
            with duplicate_files_lock:
                duplicate_files.append((input_path, original))
            update_stats('duplicates')
            return None
        hashes[phash] = input_path
    
    ext = os.path.splitext(filename)[1].lower()
    output_filename = f"{current_subclass_name}-{counter}{ext}"
    output_path = os.path.join(subfolder_output_path, output_filename)
    
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img.save(output_path)
            update_stats('saved')

            with metadata_lock:
                metadata_records.append({
                    "class": current_class_name,
                    "subclass": current_subclass_name,
                    "filename": output_filename,
                    "hash": phash,
                    "width": img.width,
                    "height": img.height,
                    "path": os.path.relpath(output_path, output_dir)
                })

        return output_filename
    except Exception as e:
        update_stats('errors')
        print(f"Error saving {input_path}: {e}")
        return None

for class_name in os.listdir(input_dir): #1/2/3
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    for subclass_name in os.listdir(class_path): #1a,1b,1c etc.
        subfolder_input_path = os.path.join(class_path, subclass_name)
        if not os.path.isdir(subfolder_input_path):
            continue

        subfolder_output_path = os.path.join(output_dir, class_name, subclass_name)
        os.makedirs(subfolder_output_path, exist_ok=True)

        # move old files in /cleaned not presented in /raw to /temp_bin
        cleaned_files = os.listdir(subfolder_output_path)
        raw_files = set(os.listdir(subfolder_input_path))
        for f in cleaned_files:
            if f not in raw_files:
                src_path = os.path.join(subfolder_output_path, f)
                relative_path = os.path.relpath(src_path, output_dir)
                dst_path = os.path.join(temp_bin_dir, relative_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.move(src_path, dst_path)
                print(f"Moved file to temp_bin: {dst_path}")
                

        # find max counter for subfolder and add new
        max_counter = 0
        for f in os.listdir(subfolder_output_path):
            match = pattern.match(f)
            if match:
                try:
                    counter = int(f.rsplit('-', 1)[1].rsplit('.', 1)[0])
                    if counter > max_counter:
                        max_counter = counter
                except:
                    pass
        counter = max_counter + 1

        print(f"Starting counter for {class_name}/{subclass_name} at: {counter}")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for filename in os.listdir(subfolder_input_path):
                if not filename.lower().endswith(valid_extensions):
                    continue
                futures.append(executor.submit(process_image, filename, subfolder_input_path, subfolder_output_path, class_name, subclass_name, counter))
                counter += 1

            for f in as_completed(futures):
                result = f.result()
                if result:
                    print(f"Saved: {os.path.join(subfolder_output_path, result)}")

# save metadata
if metadata_records:
    df = pd.DataFrame(metadata_records)
    df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to {metadata_path}")

end_time = time.time()
elapsed_time = end_time - start_time

# Summary
print("\nCleaning Summary:")
for key, value in stats.items():
    print(f"{key}: {value}")

if duplicate_files:
    print("\nDuplicate files detected:")
    for dup, original in duplicate_files:
        print(f" - {dup} (same as {original})")
else:
    print("\nNo duplicate files detected.")

print(f"\nProcessing time: {elapsed_time:.2f} seconds")
