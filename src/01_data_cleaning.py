from PIL import Image
import os
import numpy as np
import hashlib
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

start_time = time.time()
# Directories
input_dir = "dataset/raw"
output_dir = "dataset/cleaned"
temp_bin_dir = "dataset/temp_bin" # for analysing removed files

#if os.path.exists(output_dir):
#   shutil.rmtree(output_dir)

valid_extensions = (".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp", ".heic", ".png")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_bin_dir, exist_ok=True)

hashes = {}  # for duplicates

stats = {
    'total': 0,
    'saved': 0,
    'too_small': 0,
    'single_color': 0,
    'duplicates': 0,
    'errors': 0,
    'replaced': 0
}
duplicate_files = []

MIN_SIZE = 32


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
            img = img.convert("RGB") 
            if min(img.size) < MIN_SIZE:
                stats['too_small'] += 1
                return None
            if is_single_color(img):
                stats['single_color'] += 1
                return None
            phash = compute_perceptual_hash(img)
            if phash in hashes:
                duplicate_files.append((path, hashes[phash]))
                stats['duplicates'] += 1
                return None
            return phash
    except Exception as e:
        stats['errors'] += 1
        print(f"Error processing {path}: {e}")
        return None


# format check (if e.g. 1a-1.png)
pattern = re.compile(r'^([a-zA-Z0-9_]+)-\d+\.(jpg|jpeg|bmp|tiff|gif|webp|heic|png)$', re.IGNORECASE)

def process_image(filename, subfolder_input_path, subfolder_output_path, subclass_name, counter):  # nowa funkcja
    input_path = os.path.join(subfolder_input_path, filename)
    stats['total'] += 1
    phash = validate_image(input_path)
    if phash is None:
        return None
    hashes[phash] = input_path
    ext = os.path.splitext(filename)[1].lower()
    output_filename = f"{subclass_name}-{counter}{ext}"
    output_path = os.path.join(subfolder_output_path, output_filename)
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img.save(output_path)
        stats['saved'] += 1
        return output_filename
    except Exception as e:
        stats['errors'] += 1
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
                
                
        existing_files = os.listdir(subfolder_output_path)
        bad_files = []
        for f in existing_files:
            if pattern.match(f):
                output_path = os.path.join(subfolder_output_path, f)
                phash = validate_image(output_path)
                if phash is None:
                    bad_files.append(output_path)
                else:
                    hashes[phash] = output_path

        for bad_file in bad_files:
            os.remove(bad_file)
            stats['replaced'] += 1
            print(f"Removed invalid file: {bad_file}")

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
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for filename in os.listdir(subfolder_input_path):
                if not filename.lower().endswith(valid_extensions):
                    continue
                futures.append(executor.submit(process_image, filename, subfolder_input_path, subfolder_output_path, subclass_name, counter))
                counter += 1

            for f in as_completed(futures):
                result = f.result()
                if result:
                    print(f"Saved: {os.path.join(subfolder_output_path, result)}")

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