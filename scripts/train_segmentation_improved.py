#!/usr/bin/env python3
"""
Improved Segmentation Training with Better Augmentation and More Epochs
"""

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import argparse


class FloorPlanDataset(Dataset):
    """Floor Plan Segmentation Dataset"""
    
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        # Find all images
        self.image_files = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
            self.image_files.extend(glob.glob(str(self.images_dir / ext)))
        
        # Filter only those with masks
        self.valid_images = []
        for img_path in self.image_files:
            img_name = Path(img_path).name
            mask_name = Path(img_path).stem + '.png'
            mask_path = self.masks_dir / mask_name
            if mask_path.exists():
                self.valid_images.append(img_name)
        
        self.transform = transform
        print(f"  Found {len(self.valid_images)} valid image-mask pairs")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        
        # Read image
        img_path = self.images_dir / img_name
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask_name = Path(img_name).stem + '.png'
        mask_path = self.masks_dir / mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()


def get_transform(img_size=384, is_train=True):
    """Get data augmentation with more aggressive transforms"""
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined Cross Entropy + Dice Loss"""
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        return self.ce_weight * self.ce(pred, target) + self.dice_weight * self.dice(pred, target)


def calculate_iou(pred, target, num_classes):
    """Calculate IoU for each class"""
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    
    return ious


def train_improved(images_dir, masks_dir, epochs=100, batch_size=4, img_size=384, 
                   learning_rate=0.001, use_pretrained=False):
    """
    Improved training with better parameters
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        epochs: Number of epochs (default: 100)
        batch_size: Batch size (default: 4)
        img_size: Image size (default: 384)
        learning_rate: Initial learning rate (default: 0.001)
        use_pretrained: Use pretrained weights (default: False)
    """
    print("=" * 80)
    print("Improved Segmentation Training")
    print("=" * 80)
    print()
    print(f"Images directory: {images_dir}")
    print(f"Masks directory:  {masks_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Pretrained weights: {use_pretrained}")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Dataset
    print("Loading dataset...")
    dataset = FloorPlanDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=get_transform(img_size, is_train=True)
    )
    
    if len(dataset) == 0:
        print("Error: No valid image-mask pairs found!")
        return
    
    # Split train/val
    train_size = int(0.85 * len(dataset))  # Use 85% for training
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update val_dataset transform
    val_dataset.dataset.transform = get_transform(img_size, is_train=False)
    
    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    print()
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Model
    print("Creating model...")
    num_classes = 5
    encoder_weights = 'imagenet' if use_pretrained else None
    
    model = smp.Unet(
        encoder_name='resnet34',  # Larger encoder
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=None
    )
    model = model.to(device)
    print(f"Model created: U-Net with ResNet34 encoder")
    if use_pretrained:
        print("  Using ImageNet pretrained weights")
    else:
        print("  Training from scratch")
    print()
    
    # Loss and optimizer
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print()
    
    best_val_loss = float('inf')
    best_miou = 0.0
    patience_counter = 0
    early_stop_patience = 20
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': []
    }
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_ious = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate IoU
                pred = torch.argmax(outputs, dim=1)
                ious = calculate_iou(pred, masks, num_classes)
                all_ious.append(ious)
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate mean IoU
        mean_ious = np.nanmean(all_ious, axis=0)
        overall_miou = np.nanmean(mean_ious)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_miou'].append(overall_miou)
        
        # Print results
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val mIoU:   {overall_miou:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  IoU per class:")
        print(f"    background: {mean_ious[0]:.3f}")
        print(f"    wall:       {mean_ious[1]:.3f}")
        print(f"    room:       {mean_ious[2]:.3f}")
        print(f"    door_area:  {mean_ious[3]:.3f}")
        print(f"    window_area:{mean_ious[4]:.3f}")
        
        # Save best model
        is_best = False
        if overall_miou > best_miou:
            best_miou = overall_miou
            best_val_loss = avg_val_loss
            is_best = True
            patience_counter = 0
            
            output_dir = Path('models/segmentation')
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
            print(f"  [BEST MODEL SAVED] mIoU improved to {best_miou:.4f}")
        else:
            patience_counter += 1
        
        print()
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: models/segmentation/best_model.pth")
    print()
    
    # Save training history
    import json
    history_path = Path('models/segmentation/training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Improved segmentation training')
    parser.add_argument('--images', type=str, required=True,
                        help='Images directory')
    parser.add_argument('--masks', type=str, required=True,
                        help='Masks directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--img-size', type=int, default=384,
                        help='Image size (default: 384)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet pretrained weights')
    
    args = parser.parse_args()
    
    train_improved(
        args.images,
        args.masks,
        args.epochs,
        args.batch_size,
        args.img_size,
        args.lr,
        args.pretrained
    )


if __name__ == '__main__':
    main()

