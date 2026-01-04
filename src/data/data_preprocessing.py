import os
import sys
from PIL import Image
from pathlib import Path
import numpy as np

# FIXME: temporary Path hardcode
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config_loader import CONFIG
from config.logging_config import get_logger

logger = get_logger("hair_type_classifier")
logger.info("[PREPROCESSING] Starting preprocessing of images...")

MASK_DIR = CONFIG['dataset'].get('mask_dir')
RAW_DIR = CONFIG['dataset']['raw_input_dir']

def get_bbox_from_mask(mask_path, margin=0, threshold=10):

    if not Path(mask_path).exists():
        return None
    try:
        mask_pil = Image.open(mask_path).convert("L")
        mask = np.array(mask_pil)
        
        ys, xs = np.where(mask > threshold)
        if len(xs) == 0: return None 
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        width, height = mask_pil.size
        
        bbox = (
            max(0, x_min - margin),
            max(0, y_min - margin),
            min(width, x_max + margin),
            min(height, y_max + margin)
        )
        return bbox 
        
    except Exception:
        return None


def preprocess_image(input_path, target_size=None, use_masking=False, mask_dir=None):
    """
    Black background (bg removal) + Crop preprocessing
    """
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            
            if use_masking and mask_dir:
                try:
                    relative_path = Path(input_path).relative_to(CONFIG['dataset']['raw_input_dir'])
                    mask_path = Path(mask_dir) / relative_path.with_suffix(".png")
                    
                    if mask_path.exists():
                        # background removal
                        mask_pil = Image.open(mask_path).convert("L")
                        
                        # make sure mask size matches image size
                        if mask_pil.size != img.size:
                            mask_pil = mask_pil.resize(img.size, Image.NEAREST)
                        black_bg = Image.new("RGB", img.size, (0, 0, 0))
                        
                        img = Image.composite(img, black_bg, mask_pil)
                        bbox = get_bbox_from_mask(mask_path)
                        if bbox:
                            img = img.crop(bbox)
                            
                except Exception as e:
                    # if error, pass original
                    pass
            
        return img
    except Exception:
        return None

if __name__ == "__main__":
   logger.info("\n[PREPROCESSING] Module ready. Strategy: Background Removal (Black Background) + Cropping.")