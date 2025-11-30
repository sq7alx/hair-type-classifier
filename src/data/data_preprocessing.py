import os
import sys
import yaml
from PIL import Image
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config_loader import CONFIG

TARGET_SIZE = (CONFIG['dataset']['target_size_w'], CONFIG['dataset']['target_size_h'])

valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
ignore_files = [".gitkeep"]

def preprocess_image(input_path, target_size=TARGET_SIZE):
    """Load, convert to RGB, resize with black padding"""
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

        return canvas
    except Exception as e:
        print(f"Failed to preprocess {input_path}: {e}")
        return None

if __name__ == "__main__":
    print("\nPreprocessing module ready.")
    print(f"Target size set: {TARGET_SIZE}")
    print("Usage: from data_preprocessing import preprocess_image")