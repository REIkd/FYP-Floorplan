#!/usr/bin/env python3
"""
Quick Training Data Preparation - Using Existing Label Files
"""

import os
import glob
import shutil
import cv2
from pathlib import Path
import random


def parse_yolo_line(line):
    """Parse YOLO format line"""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def format_yolo_line(class_id, x, y, w, h):
    """Format YOLO line"""
    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"


def transform_flip(class_id, x, y, w, h):
    """Horizontal flip"""
    return class_id, 1.0 - x, y, w, h


def transform_rotate(class_id, x, y, w, h):
    """Rotate 90 degrees"""
    return class_id, y, 1.0 - x, h, w


def main():
    print("=" * 80)
    print("Quick Training Data Preparation")
    print("=" * 80)
    print()
    
    # Get all original label files (exclude augmented ones)
    label_files = glob.glob('data/labels_detection/*.txt')
    label_files = [f for f in label_files if 'classes' not in f and 'predefined' not in f 
                   and '_flip' not in f and '_rot90' not in f]
    
    print(f"Found {len(label_files)} original label files")
    
    # Create output directories
    output_dir = 'data/train_53'
    os.makedirs(f'{output_dir}/images/train', exist_ok=True)
    os.makedirs(f'{output_dir}/images/val', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/train', exist_ok=True)
    os.makedirs(f'{output_dir}/labels/val', exist_ok=True)
    
    # Random split into train and val sets
    random.shuffle(label_files)
    split_idx = int(len(label_files) * 0.8)
    train_labels = label_files[:split_idx]
    val_labels = label_files[split_idx:]
    
    print(f"Training set: {len(train_labels)} files")
    print(f"Validation set: {len(val_labels)} files")
    print()
    
    def process_files(label_list, split_name):
        """Process file list"""
        count = 0
        for label_path in label_list:
            label_name = Path(label_path).stem
            
            # Find corresponding image
            img_patterns = [
                f'data/images_original/{label_name}.jpg',
                f'data/images_original/{label_name}.JPG',
                f'data/images_original/{label_name}.png',
                f'data/images/{label_name}.jpg',
                f'data/images/{label_name}.JPG',
                f'data/images/{label_name}.png'
            ]
            
            img_path = None
            for pattern in img_patterns:
                if os.path.exists(pattern):
                    img_path = pattern
                    break
            
            if not img_path:
                print(f"  Skip {label_name}: Image not found")
                continue
            
            # Read image and labels
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # 1. Original
            shutil.copy2(img_path, f'{output_dir}/images/{split_name}/{label_name}_orig.jpg')
            shutil.copy2(label_path, f'{output_dir}/labels/{split_name}/{label_name}_orig.txt')
            count += 1
            
            # 2. Flip
            img_flip = cv2.flip(img, 1)
            cv2.imwrite(f'{output_dir}/images/{split_name}/{label_name}_flip.jpg', img_flip)
            with open(f'{output_dir}/labels/{split_name}/{label_name}_flip.txt', 'w') as f:
                for line in lines:
                    parsed = parse_yolo_line(line)
                    if parsed:
                        f.write(format_yolo_line(*transform_flip(*parsed)))
            count += 1
            
            # 3. Rotate
            img_rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(f'{output_dir}/images/{split_name}/{label_name}_rot.jpg', img_rot)
            with open(f'{output_dir}/labels/{split_name}/{label_name}_rot.txt', 'w') as f:
                for line in lines:
                    parsed = parse_yolo_line(line)
                    if parsed:
                        f.write(format_yolo_line(*transform_rotate(*parsed)))
            count += 1
            
        return count
    
    print("Processing training set...")
    train_count = process_files(train_labels, 'train')
    print(f"  Generated {train_count} training images")
    
    print("Processing validation set...")
    val_count = process_files(val_labels, 'val')
    print(f"  Generated {val_count} validation images")
    
    # Copy classes.txt
    shutil.copy2('data/labels_detection/classes.txt', f'{output_dir}/labels/classes.txt')
    
    # Generate config file
    config = f"""# Training Configuration
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
    print(f"  yolo train model=yolov8n.pt data={config_path} epochs=50 imgsz=640 batch=8")


if __name__ == '__main__':
    main()
