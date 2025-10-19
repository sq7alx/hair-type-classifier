from PIL import Image, ImageOps
import numpy as np
import shutil
import os

input_dir = "dataset/split"
output_dir = "dataset/processed"

os.makedirs(output_dir, exist_ok=True)

TARGET_SIZE = (224, 224)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

ignore_files = [".gitkeep"]

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)


def preprocess_image(input_path, output_path, target_size=TARGET_SIZE):
    """Load, convert to RGB, resize with black padding, normalize, save as .npy file."""
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            
            w, h = img.size
            scale = min(target_size[0]/w, target_size[1]/h)
            new_w, new_h = int(w*scale), int(h*scale)
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)

            # black padding
            canvas = Image.new("RGB", target_size, (0, 0, 0))
            top_left = ((target_size[0]-new_w)//2, (target_size[1]-new_h)//2)
            canvas.paste(img_resized, top_left)

            # Convert canvas to array and normalize
            arr = np.array(canvas).astype(np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            np.save(output_path, arr)
        return True
    except Exception as e:
        print(f"Failed to preprocess {input_path}: {e}")
        return False


for split in os.listdir(input_dir): #train/val/test
    split_input_path = os.path.join(input_dir, split)
    if not os.path.isdir(split_input_path):
        continue

    split_output_path = os.path.join(output_dir, split)
    os.makedirs(split_output_path, exist_ok=True)

    for class_name in os.listdir(split_input_path): #1/2/3
        class_input_path = os.path.join(split_input_path, class_name)
        if not os.path.isdir(class_input_path):
            continue

        class_output_path = os.path.join(split_output_path, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        for subclass_name in os.listdir(class_input_path): #1a,1b,1c etc.
            subclass_input_path = os.path.join(class_input_path, subclass_name)
            if not os.path.isdir(subclass_input_path):
                continue

            subclass_output_path = os.path.join(class_output_path, subclass_name)
            os.makedirs(subclass_output_path, exist_ok=True)

            for filename in os.listdir(subclass_input_path):
                if os.path.splitext(filename)[1].lower() not in valid_extensions:
                    continue

                input_path = os.path.join(subclass_input_path, filename)
                output_path = os.path.join(subclass_output_path, filename)

                if preprocess_image(input_path, output_path):
                    print(f"Preprocessed: {output_path}")
                # else: error already printed

print("\nPreprocessing completed", output_dir)
