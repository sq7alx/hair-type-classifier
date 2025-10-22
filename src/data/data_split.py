import os
import csv
import random

input_dir = "dataset/cleaned"
output_csv = "dataset/split/dataset_split.csv"

random.seed(1984)

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

splits = {"train": 0.7, "val": 0.15, "test": 0.15}
total_splits = sum(splits.values())

if total_splits != 1.0:
    print(f"Warning! The total splits value is {total_splits:.2f}")
    choice = input("Do you want to continue? (y/n)").lower()
    if choice != "y":
        print("Exited")
        exit(1)

rows = []

for class_name in os.listdir(input_dir):  # 1/2/3
    class_path = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    for subclass_name in os.listdir(class_path):  # 1a,1b,1c etc.
        subclass_path = os.path.join(class_path, subclass_name)
        if not os.path.isdir(subclass_path):
            continue

        images = [f for f in os.listdir(subclass_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp", ".heic", ".png"))]
        if not images:
            continue

        random.shuffle(images)

        start = 0
        total = len(images)
        for i, (split, ratio) in enumerate(splits.items()):
            if i == len(splits) - 1:
                end = total
            else:
                end = start + int(ratio * total)
            for img in images[start:end]:
                rel_path = os.path.join(class_name, subclass_name, img)
                rows.append({
                    "class": class_name,
                    "subclass": subclass_name,
                    "filename": img,
                    "split": split,
                    "path": rel_path
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
