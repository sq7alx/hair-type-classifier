import os
import shutil
import random

input_dir = "dataset/cleaned"
output_dir = "dataset/split"

random.seed(1984)

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=True)

splits = {"train": 0.7, "val": 0.15, "test": 0.15}
total_splits = sum(splits.values())

if total_splits != 1.0:
    print(f"Warning! The total splits value is {total_splits:.2f}")
    choice = input("Do you want to continue? (y/n)").lower()
    if choice != "y":
        print("Exited")
        exit(1)

for class_name in os.listdir(input_dir):  #1/2/3
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    for subclass_name in os.listdir(class_path): #1a,1b,1c etc.
        subclass_path = os.path.join(class_path, subclass_name)
        if not os.path.isdir(subclass_path):
            continue 

        images = [f for f in os.listdir(subclass_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp", ".heic", ".png"))]
        random.shuffle(images)

        start = 0
        for split, ratio in splits.items():
            split_path = os.path.join(output_dir, split, class_name, subclass_name)
            os.makedirs(split_path, exist_ok=True)
            end = start + int(ratio * len(images))
            for img in images[start:end]:
                shutil.copy(os.path.join(subclass_path, img), split_path)
            start = end

print("Data split completed")
