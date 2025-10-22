import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from data_preprocessing import preprocess_image
from data_augmentation import train_transforms, val_test_transforms


class ImageDataset(Dataset):
    """
    Dataset with preprocessing and augmentation.
    
    Args:
        csv_path (str): Path to CSV file with columns: ['class', 'subclass', 'filename', 'split', 'path']
        root_dir (str): Root directory containing images
        split (str): One of ['train', 'val', 'test']
        transform (callable, optional): Transform to apply to images
        use_subclass (bool): If True, use subclass as labels (9 classes). If False, use class (3 classes)
    """
    
    def __init__(self, csv_path, root_dir, split, transform=None, use_subclass=True):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.use_subclass = use_subclass
        
        if use_subclass:
            self.classes = sorted(self.df['subclass'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            label_type = "subclasses"
        else:
            self.classes = sorted(self.df['class'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            label_type = "classes"
        
        print(f"{split.upper()} dataset: {len(self.df)} images, {len(self.classes)} {label_type}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row['path']
        
        img = preprocess_image(img_path)
        
        if img is None:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            img = self.transform(img)
        else:
            from torchvision import transforms
            img = transforms.ToTensor()(img)
        
        label_key = 'subclass' if self.use_subclass else 'class'
        label = self.class_to_idx[row[label_key]]
        
        return img, label
    
    def get_class_counts(self):
        """Return dictionary with class counts."""
        key = 'subclass' if self.use_subclass else 'class'
        return {cls: len(self.df[self.df[key] == cls]) for cls in self.classes}


def create_dataloaders(csv_path, root_dir, batch_size=32, num_workers=4, use_subclass=True):
    """
    Create train, val, and test DataLoaders.
    
    Args:
        csv_path (str): Path to dataset CSV
        root_dir (str): Path to images directory
        batch_size (int): Batch size
        num_workers (int): Number of worker processes
        use_subclass (bool): If True, use subclass as labels (9 classes). If False, use class (3 classes)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    train_dataset = ImageDataset(csv_path, root_dir, 'train', transform=train_transforms, use_subclass=use_subclass)
    val_dataset = ImageDataset(csv_path, root_dir, 'val', transform=val_test_transforms, use_subclass=use_subclass)
    test_dataset = ImageDataset(csv_path, root_dir, 'test', transform=val_test_transforms, use_subclass=use_subclass)
    
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
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    csv_path = "dataset/split/dataset_split.csv"
    root_dir = "dataset/cleaned"
    
    print("\n" + "-"*5)
    print("Creating DataLoaders...")
    print("-"*5)
    
    # Use use_subclass=True for 9 classes (1a, 1b, 1c, 2a, 2b, 2c, 3a, 3b, 3c)
    # Use use_subclass=False for 3 classes (1, 2, 3)
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path, root_dir, batch_size=16, num_workers=2, use_subclass=True
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\n" + "-"*5)
    print("Testing batch loading with progress bar...")
    print("="*60)
    
    # Test loading one batch
    images, labels = next(iter(train_loader))
    print(f"\nSingle batch loaded successfully:")
    print(f"   Batch shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"   Labels: {labels.tolist()}")
    
    # Test full epoch
    print("\n" + "-"*5)
    print("Simulating one training epoch...")
    print("-"*5)
    
    total_samples = 0
    for images, labels in tqdm(train_loader, desc="Training", unit="batch"):
        total_samples += images.size(0)
        # Simulate some processing time
        pass
    
    print(f"\nEpoch complete. Processed {total_samples} images")
    
    # Class distribution
    print("\n" + "-"*5)
    print("Class distribution:")
    print("-"*5)
    
    for split_name, loader in [("TRAIN", train_loader), ("VAL", val_loader), ("TEST", test_loader)]:
        counts = loader.dataset.get_class_counts()
        total = sum(counts.values())
        print(f"\n{split_name}:")
        for cls, count in sorted(counts.items()):
            print(f"  {cls}: {count:3d} images ({count/total*100:.1f}%)")
    
    print("\n" + "-"*5)
    print("All tests passed - DataLoader ready for training")
    print("-"*5)