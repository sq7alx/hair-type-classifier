import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from face_parsing.model import BiSeNet
from config.config_loader import CONFIG
from config.logging_config import get_logger, setup_logger

setup_logger(
    name="hair_type_classifier",
    level="INFO",
    log_file="logs/data_segmentation.log",
    console=True,
    file=True
)
logger = get_logger("hair_type_classifier")

# params
RAW_DIR = CONFIG['dataset']['raw_input_dir']
MASK_DIR = CONFIG['dataset']['mask_dir']
BISE_MODEL_PATH = CONFIG['segmentation']['bisenet_model_path']
HAIR_CLASS_ID = CONFIG['segmentation']['hair_class_id']

os.makedirs(MASK_DIR, exist_ok=True)

# input transform
TARGET_SIZE = (448, 448)
bisenet_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

def post_process_mask(mask_np):
    if mask_np.max() == 0:
        return mask_np

    # closing
    kernel_close = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # opening
    kernel_open = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # light smoothing
    kernel_smooth = np.ones((3, 3), np.uint8)
    final_mask = cv2.erode(opened, kernel_smooth, iterations=1)
    final_mask = cv2.dilate(final_mask, kernel_smooth, iterations=1)

    return final_mask

def run_bisenet():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading BiSeNet model to {device}")

    try:
        net = BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(BISE_MODEL_PATH, map_location=device))
    except FileNotFoundError:
        logger.error(f"Model weights not found: {BISE_MODEL_PATH}")
        sys.exit(1)

    net.to(device)
    net.eval()

    all_images = []
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                all_images.append(Path(root) / f)

    logger.info(f"Found {len(all_images)} images.")
    processed_count = 0

    for img_path in tqdm(all_images, desc="BiSeNet hair masks"):
        rel = img_path.relative_to(RAW_DIR)
        out_path = Path(MASK_DIR) / rel.parent / (img_path.stem + ".png")
        os.makedirs(out_path.parent, exist_ok=True)

        if out_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size

            inp = bisenet_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                out_all = net(inp)
                out = out_all[0] if isinstance(out_all, tuple) else out_all

                out_up = F.interpolate(out, size=(orig_h, orig_w),
                                       mode='bilinear', align_corners=False)

                pred = out_up.argmax(1)
                mask_bool = (pred == HAIR_CLASS_ID).squeeze(0).cpu()

                mask_raw = (mask_bool.numpy().astype(np.uint8)) * 255
                mask_final = post_process_mask(mask_raw)

            Image.fromarray(mask_final, mode="L").save(out_path)
            processed_count += 1

        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            continue

    logger.info(f"Segmentation finished - total masks generated: {processed_count}")


if __name__ == "__main__":
    run_bisenet()
