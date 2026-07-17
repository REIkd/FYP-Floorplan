#!/usr/bin/env python3
"""
Compare segmentation baselines on the same train/val split.

Architectures (via segmentation-models-pytorch):
  - unet          : U-Net + ResNet34 (proposed)
  - deeplabv3plus : DeepLabV3+ + ResNet34 (baseline)

Results are saved to models/baseline_comparison/segmentation_results.json
"""

import argparse
import json
from pathlib import Path

import albumentations as A
import cv2
import glob
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class FloorPlanDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.valid_images = []
        for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
            for img_path in glob.glob(str(self.images_dir / ext)):
                img_name = Path(img_path).name
                mask_path = self.masks_dir / (Path(img_path).stem + '.png')
                if mask_path.exists():
                    self.valid_images.append(img_name)
        self.transform = transform

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_name = self.valid_images[idx]
        image = cv2.imread(str(self.images_dir / img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            str(self.masks_dir / (Path(img_name).stem + '.png')),
            cv2.IMREAD_GRAYSCALE,
        )
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask.long()


def get_transform(img_size, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(
            target, num_classes=pred.shape[1]
        ).permute(0, 3, 1, 2).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + 1.0) / (union + 1.0)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return 0.5 * self.ce(pred, target) + 0.5 * self.dice(pred, target)


def calculate_iou(pred, target, num_classes):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        ious.append(float('nan') if union == 0 else intersection / union)
    return ious


def build_model(architecture, num_classes, encoder_weights=None):
    common = dict(
        encoder_name='resnet34',
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=None,
    )
    if architecture == 'unet':
        return smp.Unet(**common)
    if architecture == 'deeplabv3plus':
        return smp.DeepLabV3Plus(**common)
    raise ValueError(f"Unknown architecture: {architecture}")


def train_one(architecture, images_dir, masks_dir, args, encoder_weights=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FloorPlanDataset(
        images_dir, masks_dir, transform=get_transform(args.img_size, True)
    )
    if len(dataset) == 0:
        raise RuntimeError("No image-mask pairs found.")

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_dataset.dataset.transform = get_transform(args.img_size, False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(architecture, args.num_classes, encoder_weights).to(device)
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_miou = 0.0
    best_metrics = {}
    history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
    patience = 0

    print(f"\n{'=' * 80}\nTraining {architecture.upper()} "
          f"({train_size} train / {val_size} val)\n{'=' * 80}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f'{architecture} epoch {epoch+1}'):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                pred = torch.argmax(outputs, dim=1)
                all_ious.append(calculate_iou(pred, masks, args.num_classes))

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        mean_ious = np.nanmean(all_ious, axis=0)
        overall_miou = float(np.nanmean(mean_ious))
        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_miou'].append(overall_miou)

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, mIoU={overall_miou:.4f}")

        if overall_miou > best_miou:
            best_miou = overall_miou
            best_metrics = {
                'mIoU': overall_miou,
                'wall_IoU': float(mean_ious[1]),
                'room_IoU': float(mean_ious[2]),
                'val_loss': avg_val_loss,
                'epoch': epoch + 1,
            }
            torch.save(
                model.state_dict(),
                output_dir / f'{architecture}_best.pth',
            )
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return {
        'architecture': architecture,
        'best_metrics': best_metrics,
        'history': history,
        'train_size': train_size,
        'val_size': val_size,
    }


def main():
    parser = argparse.ArgumentParser(description='Segmentation baseline comparison')
    parser.add_argument('--images', default='data/segmentation_augmented/images')
    parser.add_argument('--masks', default='data/segmentation_augmented/masks')
    parser.add_argument('--architectures', nargs='+',
                        default=['unet', 'deeplabv3plus'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--early-stop', type=int, default=20)
    parser.add_argument('--encoder-weights', default='none',
                        choices=['none', 'imagenet'],
                        help='Encoder pretraining (default: none, matching project training)')
    parser.add_argument('--output-dir', default='models/baseline_comparison')
    args = parser.parse_args()

    results = []
    enc_w = None if args.encoder_weights == 'none' else 'imagenet'
    for arch in args.architectures:
        results.append(train_one(arch, args.images, args.masks, args, enc_w))

    output_path = Path(args.output_dir) / 'segmentation_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}\nBaseline Comparison Summary\n{'=' * 80}")
    for item in results:
        m = item['best_metrics']
        print(f"{item['architecture']:15s}  mIoU={m['mIoU']*100:.2f}%  "
              f"wall={m['wall_IoU']*100:.2f}%  room={m['room_IoU']*100:.2f}%")
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
