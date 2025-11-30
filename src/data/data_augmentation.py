import sys
import random
import yaml
from torchvision import transforms
from PIL import Image, ImageEnhance
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config_loader import CONFIG

# mean and std for ImageNet
IMAGENET_MEAN = CONFIG['augmentation']['imagenet_mean']
IMAGENET_STD  = CONFIG['augmentation']['imagenet_std']

# custom transforms
class RandomRotate90:
    """Rotate the image by 90, 180, or 270 degrees randomly"""
    def __call__(self, img):
        angle = random.choice([90, 180, 270])
        return img.rotate(angle, expand=False)

class RandomVerticalFlip:
    """Flip the image vertically with a given probability"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomExposure:
    """Adjust brightness/exposure"""
    def __init__(self, factor=0.15):
        self.factor = factor
        
    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        adjustment = 1.0 + random.uniform(-self.factor, self.factor)
        return enhancer.enhance(adjustment)

# training augmentations
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotate90(),
    transforms.RandomRotation(degrees=23),
    
    RandomExposure(factor=0.15),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# validation / test augmentations
val_test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


if __name__ == "__main__":
    print("\nAugmentation transforms ready.")
    print("Train transforms:\n")
    for i, transform in enumerate(train_transforms.transforms):
        print(f"- {transform}")
    print("\nVal/Test transforms:\n", val_test_transforms)
    for i, transform in enumerate(val_test_transforms.transforms):
        print(f"- {transform}")