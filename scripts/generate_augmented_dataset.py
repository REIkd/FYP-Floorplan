#!/usr/bin/env python3
"""
Generate Augmented Dataset (Images + Labels)

Features:
    - Generate mirrored versions from original 101 images (horizontal flip)
    - Generate rotated versions from original 101 images (counterclockwise 90 degrees)
    - Generate corresponding label files simultaneously
    - Support incremental updates (only process newly labeled images)
"""

import os
import glob
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


def extract_floorplan_number(filename):
    """Extract FloorPlan number from filename"""
    try:
        name = os.path.basename(filename)
        if name.startswith('FloorPlan-'):
            parts = name.split('-')
            if len(parts) >= 2:
                number = int(parts[1])
                return number
    except:
        pass
    return None


def group_images_by_floorplan(images_dir):
    """Group images by FloorPlan number"""
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(images_dir, "*.png")))
    image_files.extend(glob.glob(os.path.join(images_dir, "*.JPG")))
    
    groups = defaultdict(list)
    
    for img_path in image_files:
        number = extract_floorplan_number(img_path)
        if number is not None:
            groups[number].append(img_path)
    
    for number in groups:
        groups[number].sort()
    
    return dict(groups)


def parse_yolo_line(line):
    """Parse one line of YOLO format annotation"""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    
    return class_id, x_center, y_center, width, height


def format_yolo_line(class_id, x_center, y_center, width, height):
    """Format as one line of YOLO format"""
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def transform_horizontal_flip(class_id, x_center, y_center, width, height):
    """
    Coordinate transformation for horizontal flip
    new_x = 1 - x
    """
    new_x = 1.0 - x_center
    new_y = y_center
    new_w = width
    new_h = height
    
    return class_id, new_x, new_y, new_w, new_h


def transform_rotate_left_90(class_id, x_center, y_center, width, height):
    """
    Coordinate transformation for counterclockwise 90-degree rotation
    new_x = y
    new_y = 1 - x
    new_width = height
    new_height = width
    """
    new_x = y_center
    new_y = 1.0 - x_center
    new_w = height
    new_h = width
    
    return class_id, new_x, new_y, new_w, new_h


def flip_image_horizontal(image):
    """Flip image horizontally"""
    return cv2.flip(image, 1)


def rotate_image_left_90(image):
    """Rotate image 90 degrees counterclockwise"""
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def generate_augmented_image_and_label(
    original_image_path,
    original_label_path,
    output_image_path,
    output_label_path,
    augmentation_type
):
    """
    Generate augmented image and corresponding label
    
    Args:
        original_image_path: Original image path
        original_label_path: Original label path
        output_image_path: Output image path
        output_label_path: Output label path
        augmentation_type: 'flip' or 'rotate'
    """
    # Read image
    image = cv2.imread(original_image_path)
    if image is None:
        print(f"  [ERROR] Cannot read image: {original_image_path}")
        return False
    
    # Apply image transformation
    if augmentation_type == 'flip':
        augmented_image = flip_image_horizontal(image)
        transform_func = transform_horizontal_flip
    elif augmentation_type == 'rotate':
        augmented_image = rotate_image_left_90(image)
        transform_func = transform_rotate_left_90
    else:
        print(f"  [ERROR] Unknown augmentation type: {augmentation_type}")
        return False
    
    # Save augmented image
    cv2.imwrite(output_image_path, augmented_image)
    
    # If label file doesn't exist, only generate image
    if not os.path.exists(original_label_path):
        print(f"  [WARN] Original label not found, only generating image")
        return True
    
    # Read original label and transform
    transformed_lines = []
    
    try:
        with open(original_label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parsed = parse_yolo_line(line)
                if parsed is None:
                    continue
                
                # Apply coordinate transformation
                transformed = transform_func(*parsed)
                transformed_line = format_yolo_line(*transformed)
                transformed_lines.append(transformed_line)
        
        # Write new label file
        with open(output_label_path, 'w', encoding='utf-8') as f:
            f.writelines(transformed_lines)
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Error processing label file: {e}")
        return False


def generate_flipped_suffix(original_name):
    """Generate filename suffix for flipped image"""
    # Add _flip identifier before hash
    # FloorPlan-1-_JPG.rf.xxxxx.jpg -> FloorPlan-1-_JPG.rf.xxxxx_flip.jpg
    stem = Path(original_name).stem
    ext = Path(original_name).suffix
    return f"{stem}_flip{ext}"


def generate_rotated_suffix(original_name):
    """Generate filename suffix for rotated image"""
    stem = Path(original_name).stem
    ext = Path(original_name).suffix
    return f"{stem}_rot90{ext}"


def generate_augmented_dataset(
    images_dir,
    labels_dir,
    output_images_dir=None,
    output_labels_dir=None,
    max_number=101,
    skip_existing=True
):
    """
    Generate complete augmented dataset
    
    Args:
        images_dir: Original images directory
        labels_dir: Original labels directory
        output_images_dir: Output images directory (default: same as original)
        output_labels_dir: Output labels directory (default: same as original)
        max_number: Maximum FloorPlan number
        skip_existing: Whether to skip existing files
    """
    
    if output_images_dir is None:
        output_images_dir = images_dir
    if output_labels_dir is None:
        output_labels_dir = labels_dir
    
    print("=" * 80)
    print("Data Augmentation Generator - Images + Labels")
    print("=" * 80)
    print()
    print(f"Original images directory: {images_dir}")
    print(f"Original labels directory: {labels_dir}")
    print(f"Output images directory: {output_images_dir}")
    print(f"Output labels directory: {output_labels_dir}")
    print()
    
    # Create output directories
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Group images
    print("Analyzing images...")
    groups = group_images_by_floorplan(images_dir)
    print(f"Found {len(groups)} FloorPlan groups")
    print()
    
    stats = {
        'flip_image': 0,
        'flip_label': 0,
        'rotate_image': 0,
        'rotate_label': 0,
        'skipped': 0
    }
    
    # Generate mirrored versions
    print("=" * 80)
    print("Step 1: Generate mirrored versions (horizontal flip)")
    print("=" * 80)
    print()
    
    for number in range(1, max_number + 1):
        if number not in groups or len(groups[number]) == 0:
            print(f"[WARN] FloorPlan-{number:3d}: Original image not found")
            continue
        
        original_image = groups[number][0]
        original_name = os.path.basename(original_image)
        original_label = os.path.join(labels_dir, Path(original_image).stem + '.txt')
        
        # Generate flipped filename
        flipped_image_name = generate_flipped_suffix(original_name)
        flipped_image_path = os.path.join(output_images_dir, flipped_image_name)
        flipped_label_path = os.path.join(output_labels_dir, Path(flipped_image_name).stem + '.txt')
        
        # Check if should skip
        if skip_existing and os.path.exists(flipped_image_path):
            print(f"[SKIP] FloorPlan-{number:3d}: Flipped image exists, skipping")
            stats['skipped'] += 1
            continue
        
        # Generate augmented data
        print(f"--> FloorPlan-{number:3d}: {original_name}")
        print(f"  Generating flipped image: {flipped_image_name}")
        
        success = generate_augmented_image_and_label(
            original_image,
            original_label,
            flipped_image_path,
            flipped_label_path,
            'flip'
        )
        
        if success:
            stats['flip_image'] += 1
            if os.path.exists(original_label):
                stats['flip_label'] += 1
                print(f"  [OK] Generated flipped label: {Path(flipped_label_path).name}")
            print()
    
    # Generate rotated versions
    print()
    print("=" * 80)
    print("Step 2: Generate rotated versions (counterclockwise 90 degrees)")
    print("=" * 80)
    print()
    
    for number in range(1, max_number + 1):
        if number not in groups or len(groups[number]) == 0:
            continue
        
        original_image = groups[number][0]
        original_name = os.path.basename(original_image)
        original_label = os.path.join(labels_dir, Path(original_image).stem + '.txt')
        
        # Generate rotated filename
        rotated_image_name = generate_rotated_suffix(original_name)
        rotated_image_path = os.path.join(output_images_dir, rotated_image_name)
        rotated_label_path = os.path.join(output_labels_dir, Path(rotated_image_name).stem + '.txt')
        
        # Check if should skip
        if skip_existing and os.path.exists(rotated_image_path):
            print(f"[SKIP] FloorPlan-{number:3d}: Rotated image exists, skipping")
            stats['skipped'] += 1
            continue
        
        # Generate augmented data
        print(f"--> FloorPlan-{number:3d}: {original_name}")
        print(f"  Generating rotated image: {rotated_image_name}")
        
        success = generate_augmented_image_and_label(
            original_image,
            original_label,
            rotated_image_path,
            rotated_label_path,
            'rotate'
        )
        
        if success:
            stats['rotate_image'] += 1
            if os.path.exists(original_label):
                stats['rotate_label'] += 1
                print(f"  [OK] Generated rotated label: {Path(rotated_label_path).name}")
            print()
    
    # Statistics
    print()
    print("=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print()
    print("Statistics:")
    print(f"  Flipped images: {stats['flip_image']} images")
    print(f"  Flipped labels: {stats['flip_label']} labels")
    print(f"  Rotated images: {stats['rotate_image']} images")
    print(f"  Rotated labels: {stats['rotate_label']} labels")
    print(f"  Skipped: {stats['skipped']} files")
    print()
    print(f"Total generated:")
    print(f"  Images: {stats['flip_image'] + stats['rotate_image']} images")
    print(f"  Labels: {stats['flip_label'] + stats['rotate_label']} labels")
    print()
    print("[INFO] Tips:")
    print("  - Augmented images and labels have been saved")
    print("  - To regenerate, use --force parameter")
    print("  - You can continue labeling original images, then rerun this script")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate augmented dataset (images + labels)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (generate augmented data from original images)
    python scripts/generate_augmented_dataset.py
    
    # Force regeneration (overwrite existing files)
    python scripts/generate_augmented_dataset.py --force
    
    # Specify custom paths
    python scripts/generate_augmented_dataset.py \\
        --images-dir data/images_original \\
        --labels-dir data/labels_detection \\
        --output-images data/images \\
        --output-labels data/labels_detection
        
Description:
    This script will:
    1. Read original images (FloorPlan-1 to FloorPlan-101)
    2. Generate horizontal flip versions (images + labels)
    3. Generate 90-degree rotation versions (images + labels)
    4. Support incremental updates (only process newly labeled images)
        """
    )
    
    parser.add_argument('--images-dir', type=str, default='data/images',
                        help='Original images directory (default: data/images)')
    parser.add_argument('--labels-dir', type=str, default='data/labels_detection',
                        help='Original labels directory (default: data/labels_detection)')
    parser.add_argument('--output-images', type=str, default=None,
                        help='Output images directory (default: same as original)')
    parser.add_argument('--output-labels', type=str, default=None,
                        help='Output labels directory (default: same as original)')
    parser.add_argument('--max-number', type=int, default=101,
                        help='Maximum FloorPlan number (default: 101)')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration, overwrite existing files')
    
    args = parser.parse_args()
    
    # Check directories
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory does not exist: {args.images_dir}")
        return
    
    if not os.path.exists(args.labels_dir):
        print(f"Warning: Labels directory does not exist: {args.labels_dir}")
        print("Will only generate augmented images, not labels")
    
    # Execute generation
    generate_augmented_dataset(
        args.images_dir,
        args.labels_dir,
        args.output_images,
        args.output_labels,
        args.max_number,
        skip_existing=not args.force
    )


if __name__ == '__main__':
    main()
