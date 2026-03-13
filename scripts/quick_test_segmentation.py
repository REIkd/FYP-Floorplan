#!/usr/bin/env python3
"""
Quick Test for Segmentation Training
Test with 19 annotated images to verify the pipeline
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
            mask_name = img_name.replace('.jpg', '.png').replace('.JPG', '.png')
            mask_path = self.masks_dir / mask_name
            if mask_path.exists():
                self.valid_images.append(img_name)
        
        self.transform = transform
        print(f"Found {len(self.valid_images)} images with masks")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        
        # Read image
        img_path = self.images_dir / img_name
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask_name = img_name.replace('.jpg', '.png').replace('.JPG', '.png')
        mask_path = self.masks_dir / mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()


def get_transform(img_size=256, is_train=True):
    """Get data augmentation"""
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


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


def train_quick_test(images_dir, masks_dir, epochs=20, batch_size=2, img_size=256):
    """
    Quick training test
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        epochs: Number of epochs (default: 20)
        batch_size: Batch size (default: 2)
        img_size: Image size (default: 256)
    """
    print("=" * 80)
    print("Quick Segmentation Training Test")
    print("=" * 80)
    print()
    print(f"Images directory: {images_dir}")
    print(f"Masks directory:  {masks_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}x{img_size}")
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
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")
    print()
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Model
    print("Creating model...")
    num_classes = 5  # background, wall, room, door_area, window_area
    model = smp.Unet(
        encoder_name='resnet18',  # Small encoder for quick test
        encoder_weights=None,  # Train from scratch for quick test
        classes=num_classes,
        activation=None
    )
    model = model.to(device)
    print("Model created: U-Net with ResNet18 encoder (training from scratch)")
    print()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print()
    
    best_val_loss = float('inf')
    
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
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val mIoU:   {overall_miou:.4f}")
        print(f"  IoU per class: background={mean_ious[0]:.3f}, wall={mean_ious[1]:.3f}, room={mean_ious[2]:.3f}")
        print()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_dir = Path('models/segmentation')
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / 'quick_test_model.pth')
            print(f"  [BEST] Model saved!")
            print()
    
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: models/segmentation/quick_test_model.pth")
    print()
    print("Next steps:")
    print("  1. Check model quality by visual inspection")
    print("  2. If quality is acceptable, annotate more images (target: 30-50)")
    print("  3. Train with more data for better performance")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick test for segmentation training')
    parser.add_argument('--images-dir', type=str, default='data/images_original',
                        help='Images directory')
    parser.add_argument('--masks-dir', type=str, default='data/labels_segmentation/masks',
                        help='Masks directory')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Image size (default: 256)')
    
    args = parser.parse_args()
    
    train_quick_test(
        args.images_dir,
        args.masks_dir,
        args.epochs,
        args.batch_size,
        args.img_size
    )


if __name__ == '__main__':
    main()

