import os
import sys
import argparse
import subprocess
import shutil
import logging
import torch
import pandas as pd
import yaml
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from src.data.data_preprocessing import preprocess_image
from src.data.data_augmentation import train_transforms, val_test_transforms

from config.logging_config import get_logger, setup_logger

setup_logger(
    name="hair_type_classifier",
    level=logging.INFO,
    log_file="logs/data_pipeline.log",
    console=True,
    file=True
)

logger = get_logger(__name__)

def load_config(config_path="config/config.yaml"):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")
    with open(config_file, "r") as f:
        return yaml.safe_load(f)
try:
    CONFIG = load_config()
    logger.info("Configuration loaded successfully")
except FileNotFoundError as e:
    logger.error(e)
    sys.exit(1)

class ImageDataset(Dataset):
    """
    Dataset with preprocessing and augmentation.
    
    Args:
        csv_path (str): Path to CSV file with columns: ['class', 'subclass', 'filename', 'split', 'path']
        root_dir (str): Root directory containing images
        split (str): One of ['train', 'val', 'test']
        transform (callable, optional): Transform to apply to images
        use_subclass (bool): If True, use subclass as labels (9 classes). If False, use class (3 classes)
        use_masking (bool): If True, apply hair mask during preprocessing.
    """
    
    def __init__(self, csv_path, root_dir, split, transform=None, use_subclass=True, use_masking=False, mask_dir=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.use_subclass = use_subclass
        self.use_masking = use_masking
        self.mask_dir = Path(mask_dir) if mask_dir else None
        
        if use_subclass:
            self.classes = sorted(self.df['subclass'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            label_type = "subclasses"
        else:
            self.classes = sorted(self.df['class'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            label_type = "classes"
        
        logger.info(f"{split.upper()} dataset: {len(self.df)} images, {len(self.classes)} {label_type} (Masking: {use_masking})")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row['path']
        
        # getting use_masking for preprocess_image
        img = preprocess_image(img_path, use_masking=self.use_masking, mask_dir=self.mask_dir)
        
        if img is None:
            logger.warning(f"Failed to load image: {img_path}, using blank image.")
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        label_key = 'subclass' if self.use_subclass else 'class'
        label = self.class_to_idx[row[label_key]]
        
        return img, label
    
    def get_class_counts(self):
        """Return dictionary with class counts."""
        key = 'subclass' if self.use_subclass else 'class'
        return {cls: len(self.df[self.df[key] == cls]) for cls in self.classes}


def create_dataloaders(batch_size, num_workers, use_subclass=False, use_masking=False):
    """
    Create train, val, and test DataLoaders.
    """
    csv_path = CONFIG['dataset']['split_output_csv']
    root_dir = CONFIG['dataset']['raw_input_dir']
    mask_dir = CONFIG['dataset']['mask_dir']
    
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file does not exist: {csv_path}. Please run the data cleaning and splitting first.")
        return None, None, None
    
    logger.info(f"Loading datasets from CSV: {csv_path}")
    logger.info(f"Using root directory: {root_dir}")
    
    train_dataset = ImageDataset(csv_path, root_dir, 'train', transform=train_transforms, use_subclass=use_subclass, use_masking=use_masking, mask_dir=mask_dir)
    val_dataset = ImageDataset(csv_path, root_dir, 'val', transform=val_test_transforms, use_subclass=use_subclass, use_masking=use_masking, mask_dir=mask_dir)
    test_dataset = ImageDataset(csv_path, root_dir, 'test', transform=val_test_transforms, use_subclass=use_subclass, use_masking=use_masking, mask_dir=mask_dir)
    
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    logger.info("DataLoaders created successfully.")
    return train_loader, val_loader, test_loader


class PipelineManager:
    """
    Executes data pipeline stages
    """

    def __init__(self, args):
        self.args = args
        self.cleaned_dir = Path(CONFIG['dataset']['cleaned_output_dir'])
        self.split_csv = Path(CONFIG['dataset']['split_output_csv'])

        self.stages = [
            ("Data Cleaning", not args.skip_cleaning, "src/data/data_cleaning.py"),
            ("Data Splitting", not args.skip_splitting, "src/data/data_split.py"),
            ("Preprocessing", not args.skip_preprocessing, "src/data/data_preprocessing.py"),
            ("Augmentation", not args.skip_augmentation, "src/data/data_augmentation.py"),
        ]

    def validate_inputs(self):
        """Validate that necessary files/directories exist based on skip flags."""

        # skip_cleaning requires cleaned_dir present
        if self.args.skip_cleaning:
            if not self.cleaned_dir.exists() or not any(self.cleaned_dir.iterdir()):
                raise RuntimeError(
                    f"Cannot proceed: --skip_cleaning requires '{self.cleaned_dir}' to exist and be non-empty."
                )

        # skip_splitting requires CSV present and valid
        if self.args.skip_splitting:
            self._validate_split_csv()

    def prepare_clean_directory(self):
        if self.args.skip_cleaning:
            return

        if self.cleaned_dir.exists() and any(self.cleaned_dir.iterdir()):
            logger.warning(f"Directory '{self.cleaned_dir}' exists and is not empty.")
            response = input("All content will be deleted. Do you want to continue? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Operation cancelled. Use --skip_cleaning flag to bypass this step.")
                sys.exit(0)
            logger.info(f"Deleting contents of '{self.cleaned_dir}'...")
            shutil.rmtree(self.cleaned_dir)
            logger.info(f"Directory '{self.cleaned_dir}' is now clean and ready.")

        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

    def _validate_split_csv(self):
        """Ensure CSV exists and contains train/val/test."""
        if not self.split_csv.exists():
            raise RuntimeError(f"Missing required CSV: {self.split_csv}")

        df = pd.read_csv(self.split_csv)
        if df.empty:
            raise RuntimeError(f"{self.split_csv} is empty.")

        if not {'train', 'val', 'test'}.issubset(df['split'].unique()):
            raise RuntimeError(f"{self.split_csv} is missing required splits (train, val, test).")

    def run_stages(self):
        """Execute pipeline scripts in sequence."""
        logger.info("=" * 60)
        logger.info("Running Full Data Pipeline")
        logger.info("=" * 60)

        for name, should_run, script in self.stages:
            if not should_run:
                logger.info(f"[SKIP] Skipping: {name}")
                continue

            logger.info(f"[STAGE] Starting: {name}")
            result = subprocess.run([sys.executable, script], check=True)

            if result.returncode != 0:
                logger.error(f"Stage failed: {name}")
                logger.error(result.stderr)
                raise RuntimeError(f"Stage failed: {name}")

            logger.info(f"[SUCCESS] Completed: {name}")

        logger.info("=" * 60)
        logger.info("Pipeline execution completed.")
        logger.info("=" * 60)


# test and valideation
def test_dataloaders(train_loader, val_loader, test_loader):
    """test DataLoaders and display statistics"""
    
    logger.info(f"DataLoader batch counts:")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    
    logger.info("-" * 40)
    logger.info("Testing batch loading...")
    logger.info("-" * 40)
    
    # test loading one batch
    images, labels = next(iter(train_loader))
    logger.info("Single batch loaded successfully:")
    logger.info(f"   Batch shape: {images.shape}")
    logger.info(f"   Labels shape: {labels.shape}")
    logger.info(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
    logger.info(f"   Labels sample: {labels.tolist()}")
    
    
    # class distribution
    logger.info("-" * 40)
    logger.info("Class distribution per split:")
    logger.info("-" * 40)
    
    for split_name, loader in [("TRAIN", train_loader), ("VAL", val_loader), ("TEST", test_loader)]:
        counts = loader.dataset.get_class_counts()
        total = sum(counts.values())
        logger.info(f"\n{split_name}:")
        for cls, count in sorted(counts.items()):
            logger.info(f"  {cls}: {count:3d} images ({count/total*100:.1f}%)")
    
    logger.info("=" * 60)
    logger.info("All tests passed - DataLoader ready for training")

# saving sample images for verification
def save_sample_images(dataset, output_dir, count=10, prefix="processed"):
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    IMAGENET_MEAN = CONFIG['augmentation']['imagenet_mean']
    IMAGENET_STD = CONFIG['augmentation']['imagenet_std']
    
    mean_tensor = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_tensor = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    logger.info("-" * 60)
    logger.info(f"Saving {count} sample images to: {output_path}")
    logger.info("-" * 60)

    for i in tqdm(range(min(count, len(dataset))), desc=f"Saving {prefix} samples"):
        try:
            image_tensor, label = dataset[i]
            # denormalize from ImageNet back to [0, 1]
            image_unnormalized = image_tensor * std_tensor + mean_tensor
            image_np = image_unnormalized.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            
            image_np = np.clip(image_np, 0, 255) 
            img_pil = Image.fromarray(image_np)

            label_name = dataset.classes[label]
            
            save_name = f"{prefix}_{i}_{label_name}.png"
            img_pil.save(output_path / save_name)
            
        except Exception as e:
            logger.error(f"Failed to save sample image {i}: {e}")
            continue

def run_pipeline():
    parser = argparse.ArgumentParser(description="Run full data pipeline with optional stages")
    parser.add_argument('--skip_cleaning', action='store_true', help='Skip data cleaning step')
    parser.add_argument('--skip_splitting', action='store_true', help='Skip data splitting step')
    parser.add_argument('--skip_preprocessing', action='store_true', help='Skip data preprocessing step')
    parser.add_argument('--skip_augmentation', action='store_true', help='Skip data augmentation step')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    args = parser.parse_args()

    # log level setup
    main_logger = logging.getLogger("hair_type_classifier")
    main_logger.setLevel(getattr(logging, args.log_level))
    logger.info(f"Log level set to: {args.log_level}")

    pm = PipelineManager(args)
    pm.validate_inputs()
    pm.prepare_clean_directory()
    pm.run_stages()

    return args

if __name__ == "__main__":
    args = run_pipeline()

    batch_size_conf = CONFIG['defaults']['batch_size']
    num_workers_conf = CONFIG['defaults']['num_workers']
    use_subclass_conf = CONFIG['defaults']['use_subclass']
    use_masking_conf = CONFIG['defaults'].get('use_masking', False) 
    
    logger.info("=" * 60)
    logger.info("Creating DataLoaders...")
    logger.info("=" * 60)

    try:
        loaders = create_dataloaders(
            batch_size=batch_size_conf,
            num_workers=num_workers_conf,
            use_subclass=use_subclass_conf,
            use_masking=use_masking_conf
        )

        train_loader, val_loader, test_loader = loaders

        test_dataloaders(train_loader, val_loader, test_loader)
            
        # saving masked samples for visualisation
        OUTPUT_FOLDER = Path("dataset/processed_samples")
            
        logger.info(f"Saving sample images to verify preprocessing/augmentation to {OUTPUT_FOLDER}")
            
        # training samples (with augmentation)
        save_sample_images(train_loader.dataset, OUTPUT_FOLDER / "train", count=10, prefix="train_aug")  
        # validation samples (w/o augmentation)
        save_sample_images(val_loader.dataset, OUTPUT_FOLDER / "val", count=10, prefix="val_norm")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during DataLoader creation: {e}")
        sys.exit(1)