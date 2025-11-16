import sys
import os
import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.run_pipeline import create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train a computer vision classification model")
    
    parser.add_argument("--csv-path", type=str, default="dataset/split/dataset_split.csv", help="Path to dataset CSV file (default: dataset/split/dataset_split.csv)")
    parser.add_argument("--root-dir", type=str, default="dataset/cleaned", help="Path to dataset root directory (default: dataset/cleaned)")
    parser.add_argument("--use-subclass", action="store_true", help="Use subclass labels (9 classes) instead of main class (3 classes)")
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze pretrained backbone except final layer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def get_model(arch, num_classes, freeze_backbone=False):
    
    if arch == "resnet18":
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet34":
        model = models.resnet34(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading datasets")
    print(f"CSV path: {args.csv_path}")
    print(f"Root directory: {args.root_dir}")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_subclass=args.use_subclass
    )
    
    if train_loader is None:
        print("Error loading datasets. Exiting.")
        return

    num_classes = len(train_loader.dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Dataset sizes: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}")
    
    # model initialization
    model = get_model(args.arch, num_classes, args.freeze_backbone).to(device)
    print(f"Initialized {args.arch} with {num_classes} output classes")
    if args.freeze_backbone:
        print("Backbone frozen, only final layer will be trained")
    
    # loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    # history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.output, f"best_model_{args.arch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'args': vars(args)
            }, best_model_path)
            print(f" -> New best model saved with accuracy: {best_val_acc:.2f}%")
        
        scheduler.step()
    
    # last model saving
    final_model_path = os.path.join(args.output, f"final_model_{args.arch}.pth")
    torch.save(model.state_dict(), final_model_path)

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {os.path.join(args.output, f'best_model_{args.arch}.pth')}")
    print(f"Final model saved to: {final_model_path}")
    
    print("\nEvaluating on test set...")
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    test_acc = 100. * correct_test / total_test
    print(f"Final test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()