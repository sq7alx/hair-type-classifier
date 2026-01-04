import sys
import random
import yaml
from torchvision import transforms
from pathlib import Path

# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

# FIXME: temporary Path hardcode
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config_loader import CONFIG
from config.logging_config import get_logger, setup_logger

setup_logger(
    name="hair_type_classifier",
    level="INFO",
    log_file="logs/data_augmentation.log",
    console=True,
    file=True
)
logger = get_logger("hair_type_classifier")

# mean and std for ImageNet
IMAGENET_MEAN = CONFIG['augmentation']['imagenet_mean']
IMAGENET_STD  = CONFIG['augmentation']['imagenet_std']

TARGET_SIZE = (
    CONFIG['dataset']['target_size_w'],
    CONFIG['dataset']['target_size_h']
)

# training augmentations
train_transforms = transforms.Compose([
    transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15, fill=0),
    
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.05
    ),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    
    transforms.RandomErasing(
        p=0.3,
        scale=(0.02, 0.15),
        ratio=(0.3, 3.3),
        value='random'
    ),
])


# validation / test augmentations
val_test_transforms = transforms.Compose([
    transforms.Resize(TARGET_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

if __name__ == "__main__":
    logger.info(f"[AUGMENTATION] Target size: {TARGET_SIZE}")
    logger.info("[AUGMENTATION] Train transforms:")
    for t in train_transforms.transforms:
        logger.info(f"  - {t.__class__.__name__}")

    logger.info("[AUGMENTATION] Val/Test transforms:")
    for t in val_test_transforms.transforms:
        logger.info(f"  - {t.__class__.__name__}")