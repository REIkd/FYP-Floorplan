"""
Dataset Preparation Tool
Split dataset into training, validation and test sets
"""

import os
import random
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(images_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Split dataset
    
    Args:
        images_dir: Images directory
        output_dir: Output directory
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    """
    random.seed(seed)
    
    # Get all images
    images_path = Path(images_dir)
    image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
    image_files = [f.name for f in image_files]
    
    print(f"Found {len(image_files)} images")
    
    # First split: separate test set
    train_val, test = train_test_split(
        image_files, 
        test_size=test_ratio, 
        random_state=seed
    )
    
    # Second split: separate validation set from remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        random_state=seed
    )
    
    print(f"Training set: {len(train)} images")
    print(f"Validation set: {len(val)} images")
    print(f"Test set: {len(test)} images")
    
    # Save split results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save file lists
    with open(output_path / 'train.txt', 'w') as f:
        f.write('\n'.join(train))
    
    with open(output_path / 'val.txt', 'w') as f:
        f.write('\n'.join(val))
    
    with open(output_path / 'test.txt', 'w') as f:
        f.write('\n'.join(test))
    
    print(f"\nDataset split complete, results saved to: {output_path}")
    
    return train, val, test

def check_annotations(images_dir, labels_dir, file_list):
    """
    Check if annotation files are complete
    
    Args:
        images_dir: Images directory
        labels_dir: Labels directory
        file_list: File list
    """
    missing = []
    
    for img_file in file_list:
        # Check corresponding label file
        label_file = Path(img_file).stem + '.txt'
        label_path = Path(labels_dir) / label_file
        
        if not label_path.exists():
            missing.append(img_file)
    
    if missing:
        print(f"\nWarning: {len(missing)} files are missing labels:")
        for f in missing[:10]:  # Show first 10 only
            print(f"  - {f}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    else:
        print("[OK] All files have corresponding labels")
    
    return missing

def create_yolo_yaml(dataset_root, output_path, class_names):
    """
    Create YOLO format dataset configuration file
    
    Args:
        dataset_root: Dataset root directory
        output_path: Output YAML file path
        class_names: List of class names
    """
    yaml_content = f"""# YOLO Dataset Configuration
path: {dataset_root}
train: splits/train.txt
val: splits/val.txt
test: splits/test.txt

# Classes
nc: {len(class_names)}
names: {class_names}
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"YOLO config file created: {output_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('--images-dir', type=str, default='data/images',
                       help='Images directory')
    parser.add_argument('--output-dir', type=str, default='data/splits',
                       help='Output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--check-labels', type=str, default=None,
                       help='Check labels directory')
    
    args = parser.parse_args()
    
    # Split dataset
    train, val, test = split_dataset(
        args.images_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    # Check labels
    if args.check_labels:
        print("\nChecking training set labels...")
        check_annotations(args.images_dir, args.check_labels, train)
        print("\nChecking validation set labels...")
        check_annotations(args.images_dir, args.check_labels, val)
        print("\nChecking test set labels...")
        check_annotations(args.images_dir, args.check_labels, test)

if __name__ == '__main__':
    main()
