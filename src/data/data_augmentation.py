from torchvision import transforms
import random
from PIL import Image, ImageEnhance

# Mean and Std for ImageNet (standard normalization)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# CUSTOM TRANSFORMS
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

# TRAINING AUGMENTATIONS
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotate90(),
    transforms.RandomRotation(degrees=23),
    
    RandomExposure(factor=0.15),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


# VALIDATION / TEST AUGMENTATIONS
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