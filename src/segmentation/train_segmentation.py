"""
Room Semantic Segmentation Model Training Script
Using U-Net or other segmentation models for room and wall segmentation
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FloorPlanDataset(Dataset):
    """Floor Plan Dataset"""
    
    def __init__(self, images_dir, masks_dir, image_list, transform=None):
        """
        Initialize dataset
        
        Args:
            images_dir: Images directory
            masks_dir: Masks directory
            image_list: List of image filenames
            transform: Data augmentation
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_list = image_list
        self.transform = transform
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # Read image
        img_name = self.image_list[idx]
        img_path = self.images_dir / img_name
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = self.masks_dir / mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Data augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()

def get_transforms(config, is_train=True):
    """Get data augmentation configuration"""
    img_size = config['data']['img_size']
    
    if is_train:
        aug = config['augmentation']
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.HorizontalFlip(p=aug.get('horizontal_flip', 0.5)),
            A.Rotate(limit=aug.get('rotate', 15), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug.get('brightness', 0.2),
                contrast_limit=aug.get('contrast', 0.2),
                p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def create_model(config):
    """Create segmentation model"""
    model_cfg = config['model']
    num_classes = config['num_classes']
    
    model_name = model_cfg['name'].lower()
    
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=model_cfg['encoder'],
            encoder_weights=model_cfg.get('encoder_weights', 'imagenet'),
            classes=num_classes,
            activation=model_cfg.get('activation', None)
        )
    elif model_name == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=model_cfg['encoder'],
            encoder_weights=model_cfg.get('encoder_weights', 'imagenet'),
            classes=num_classes,
            activation=model_cfg.get('activation', None)
        )
    elif model_name == 'fpn':
        model = smp.FPN(
            encoder_name=model_cfg['encoder'],
            encoder_weights=model_cfg.get('encoder_weights', 'imagenet'),
            classes=num_classes,
            activation=model_cfg.get('activation', None)
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_room_segmentation(config_path):
    """Train room segmentation model"""
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Loss function
    loss_name = config['training'].get('loss', 'dice')
    if loss_name == 'dice':
        criterion = smp.losses.DiceLoss(mode='multiclass')
    elif loss_name == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif loss_name == 'focal':
        criterion = smp.losses.FocalLoss(mode='multiclass')
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # TODO: Load dataset (need annotated data prepared)
    # train_dataset = FloorPlanDataset(...)
    # val_dataset = FloorPlanDataset(...)
    # train_loader = DataLoader(train_dataset, ...)
    # val_loader = DataLoader(val_dataset, ...)
    
    print("Note: Annotated data must be prepared before training!")
    print("Please use LabelMe or similar tools to annotate room and wall masks")
    
    # Training loop
    epochs = config['training']['epochs']
    best_loss = float('inf')
    
    # for epoch in range(epochs):
    #     train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    #     val_loss = validate(model, val_loader, criterion, device)
    #     scheduler.step(val_loss)
    #     
    #     if val_loss < best_loss:
    #         best_loss = val_loss
    #         torch.save(model.state_dict(), 'models/segmentation/best_model.pth')

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train room segmentation model')
    parser.add_argument('--config', type=str,
                       default='config/room_segmentation.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    train_room_segmentation(args.config)

if __name__ == '__main__':
    main()
