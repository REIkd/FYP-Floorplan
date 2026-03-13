#!/usr/bin/env python3
"""
Prepare training dataset for 90 labeled images
"""

import os
import glob
import shutil
import random
from pathlib import Path

def main():
    print("=" * 80)
    print("Preparing Training Dataset - 90 Images")
    print("=" * 80)
    print()
    
    # Get all label files (including augmented)
    label_files = glob.glob('data/labels_detection/*.txt')
    label_files = [f for f in label_files if 'classes' not in f and 'predefined' not in f]
    
    print(f"Found {len(label_files)} label files")
    
    # Create output directories
    output_dir = 'data/train_90'
    os.makedirs(f'{output_dir}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/images/val', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/train', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/val', exist_ok=True)
    
    # Random split into train and val
    random.shuffle(label_files)
    split_idx = int(len(label_files) * 0.8)
    train_labels = label_files[:split_idx]
    val_labels = label_files[split_idx:]
    
    print(f"Training set: {len(train_labels)} files")
    print(f"Validation set: {len(val_labels)} files")
    print()
    
    def copy_files(label_list, split_name):
        """Copy image and label files"""
        count = 0
        for label_path in label_list:
            label_name = Path(label_path).stem
            
            # Find corresponding image
            img_patterns = [
                f'data/images/{label_name}.jpg',
                f'data/images/{label_name}.JPG',
                f'data/images/{label_name}.png',
                f'data/images_original/{label_name}.jpg',
                f'data/images_original/{label_name}.JPG',
            ]
            
            img_path = None
            for pattern in img_patterns:
                if os.path.exists(pattern):
                    img_path = pattern
                    break
            
            if not img_path:
                continue
            
            # Copy image and label
            shutil.copy2(img_path, f'{output_dir}/images/{split_name}/{label_name}.jpg')
            shutil.copy2(label_path, f'{output_dir}/labels/{split_name}/{label_name}.txt')
            count += 1
            
        return count
    
    print("Copying training set...")
    train_count = copy_files(train_labels, 'train')
    print(f"  Copied {train_count} training images")
    
    print("Copying validation set...")
    val_count = copy_files(val_labels, 'val')
    print(f"  Copied {val_count} validation images")
    
    # Copy classes.txt
    shutil.copy2('data/labels_detection/classes.txt', f'{output_dir}/labels/classes.txt')
    
    # Generate config file
    config = f"""# Training Configuration - 90 Images Round 2
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: door
  1: window
  2: table
  3: chair
  4: bed
  5: sofa
  6: toilet
  7: sink
  8: bathtub
  9: stove
  10: refrigerator
  11: wardrobe
  12: tv
  13: desk
  14: washingmachine
  15: loadbearing_wall
  16: aircondition
  17: cupboard

nc: 18
"""
    
    config_path = f'{output_dir}/data.yaml'
    with open(config_path, 'w') as f:
        f.write(config)
    
    print()
    print("=" * 80)
    print("Complete!")
    print("=" * 80)
    print(f"Training images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Total: {train_count + val_count}")
    print()
    print(f"Config file: {config_path}")
    print()
    print("Start training:")
    print(f"  yolo train model=yolov8n.pt data={config_path} epochs=100 imgsz=640 batch=8 device=cpu")
    print()
    print("Or use larger model for better results:")
    print(f"  yolo train model=yolov8s.pt data={config_path} epochs=100 imgsz=640 batch=8 device=cpu")


if __name__ == '__main__':
    main()

