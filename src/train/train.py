import sys
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

# FIXME: temporary path hack
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_pipeline import create_dataloaders
from config.config_loader import CONFIG

DEFAULT_CSV_PATH = CONFIG['dataset']['split_output_csv']
DEFAULT_ROOT_DIR = CONFIG['dataset']['raw_input_dir']

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for H0 (Main) and H1/H2/H3 (Subclass) models")
    parser.add_argument("--parent-class", type=str, default=None, choices=["1", "2", "3"],
                        help="If set, trains a specialist model (H1/H2/H3) for subtypes of this parent(e.g. \"--parent-class 1\" means learning H1 model for 3 classes 1a/1b/1c)."
                             "If NOT set, trains the main H0 model.")
    
    parser.add_argument("--arch", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze pretrained backbone except final layer")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait before early stopping.")
    return parser.parse_args()

def get_model(arch, num_classes, freeze_backbone=False):
    
    if arch == "resnet18":
        model = models.resnet18(weights='IMAGENET1K_V1')
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3), 
            nn.Linear(in_features, num_classes)
        )
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
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in pbar:
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
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    
    # metrics calculation (accuracy and F1-score)
    correct = (np.array(all_preds) == np.array(all_labels)).sum()
    epoch_acc = 100. * correct / len(all_labels)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1


def main():
    args = parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            
        np.random.seed(args.seed)
        
        logger.info(f" Seed fixed: {args.seed}")
    else:
        logger.info("Seed: random")    
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    if args.parent_class is None:
        run_mode = "H0_Main"
        use_subclass_flag = False
        target_type_filter = None
        default_output_dir = os.path.join(args.output, "h0_main")
        logger.info(f"Mode {run_mode} - Level 0 (H0 model with main classes)")
        
    else:
        run_mode = f"H{args.parent_class}_Subclass"
        use_subclass_flag = True
        target_type_filter = int(args.parent_class)
        default_output_dir = os.path.join(args.output, f"h{target_type_filter}_subclass")
        logger.info(f"Mode {run_mode} - Level 1 (H{target_type_filter} model with subclasses a/b/c)")
    
    os.makedirs(default_output_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_subclass=use_subclass_flag,
        use_masking=CONFIG['defaults']['use_masking'], 
        parent_class=target_type_filter
    )
    
    if train_loader is None or len(train_loader.dataset) == 0:
        logger.error("No data available")
        return

    classes = train_loader.dataset.classes
    num_classes = len(classes)
    logger.info(f"Detected {num_classes} classes: {classes}")
        
    # model initialization
    model = get_model(args.arch, num_classes).to(device)
    
    # weight decay based on run mode (H0 has bigger dataset than H1/H2/H3)
    wd = 1e-4 if run_mode == "H0_Main" else 1e-2
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )
    
    logger.info(f"Starting training: {args.arch}, {num_classes} classes | Device: {device} | Output: {default_output_dir}")
    
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate_epoch(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1:02d}/{args.epochs} | "
                    f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                    f"Val F1: {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_acc)
                
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            
            filename = f"best_model_{args.arch}_{run_mode}.pth"
            best_model_path = os.path.join(default_output_dir, filename)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'classes': classes,
                'args': vars(args),
                'mode': run_mode
            }, best_model_path)
            logger.info(f" New best model saved: {best_val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # test evaluation
    logger.info("\nEvaluating on test set...")
    filename = f"best_model_{args.arch}_{run_mode}.pth"
    best_model_path = os.path.join(default_output_dir, filename)
    
    if os.path.exists(best_model_path):
        
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            loop = tqdm(test_loader, desc="Final Testing", leave=False)
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
        test_acc = 100. * (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
            
        logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
        report = classification_report(all_labels, all_preds, target_names=test_loader.dataset.classes)
        logger.info("\nDetailed Classification Report:\n" + report)
    else:
        logger.error("No best model found to evaluate.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)