#!/usr/bin/env python3
"""
Extract original 101 images from 303 images

Features:
    - Identify FloorPlan-1 to FloorPlan-101 images
    - Take the first image from each group as the original
    - Copy to specified directory
"""

import os
import glob
import shutil
from pathlib import Path
from collections import defaultdict


def extract_floorplan_number(filename):
    """
    Extract FloorPlan number from filename
    
    Example: FloorPlan-1-_JPG.rf.xxxxx.jpg -> 1
             FloorPlan-101-_jpg.rf.xxxxx.jpg -> 101
    """
    try:
        # Extract number after FloorPlan-
        name = os.path.basename(filename)
        if name.startswith('FloorPlan-'):
            # Find number part after first '-'
            parts = name.split('-')
            if len(parts) >= 2:
                number_str = parts[1]
                # Extract pure number
                number = int(number_str)
                return number
    except:
        pass
    return None


def group_images_by_floorplan(images_dir):
    """
    Group images by FloorPlan number
    
    Returns: {number: [list of image paths]}
    """
    # Get all images
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(images_dir, "*.png")))
    image_files.extend(glob.glob(os.path.join(images_dir, "*.JPG")))
    
    # Group by FloorPlan number
    groups = defaultdict(list)
    
    for img_path in image_files:
        number = extract_floorplan_number(img_path)
        if number is not None:
            groups[number].append(img_path)
    
    # Sort images within each group
    for number in groups:
        groups[number].sort()
    
    return dict(groups)


def extract_original_images(images_dir, output_dir, max_number=101, dry_run=False):
    """
    Extract original images (first image from each group)
    
    Args:
        images_dir: Original images directory
        output_dir: Output directory
        max_number: Maximum FloorPlan number (default: 101)
        dry_run: If True, only show files to process without actually copying
    """
    print("=" * 80)
    print("Extract Original Images Tool")
    print("=" * 80)
    print()
    
    # Group images
    print(f"Analyzing images directory: {images_dir}")
    groups = group_images_by_floorplan(images_dir)
    
    print(f"Found {len(groups)} FloorPlan groups")
    print()
    
    # Create output directory
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        print(f"[Preview Mode] Output directory: {output_dir}")
    print()
    
    # Extract first image from each group
    print("=" * 80)
    print("Extracting Original Images")
    print("=" * 80)
    print()
    
    extracted = []
    skipped = []
    
    for number in range(1, max_number + 1):
        if number in groups:
            images = groups[number]
            if len(images) > 0:
                # Take first image as original
                original_image = images[0]
                original_name = os.path.basename(original_image)
                output_path = os.path.join(output_dir, original_name)
                
                if dry_run:
                    print(f"[Preview] FloorPlan-{number:3d}: {original_name}")
                else:
                    # Copy file
                    shutil.copy2(original_image, output_path)
                    print(f"[OK] FloorPlan-{number:3d}: {original_name}")
                
                extracted.append(number)
                
                # Show other images in group (augmented versions)
                if len(images) > 1:
                    for aug_img in images[1:]:
                        aug_name = os.path.basename(aug_img)
                        print(f"  └─ Augmented version: {aug_name}")
        else:
            print(f"[WARN] FloorPlan-{number:3d}: Not found")
            skipped.append(number)
    
    # Statistics
    print()
    print("=" * 80)
    print("Extraction Complete")
    print("=" * 80)
    print(f"Successfully extracted: {len(extracted)} original images")
    print(f"Missing: {len(skipped)} images")
    
    if skipped:
        print()
        print("Missing FloorPlan numbers:")
        print(f"  {', '.join(map(str, skipped[:20]))}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")
    
    print()
    if not dry_run:
        print(f"[OK] Original images saved to: {output_dir}")
    else:
        print("[INFO] This is preview mode, files were not actually copied")
        print("   To execute, remove --dry-run parameter")
    
    return extracted, skipped


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract original 101 images from 303 images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview images to extract (without actually copying)
    python scripts/extract_original_images.py --dry-run
    
    # Extract to default directory (data/images_original)
    python scripts/extract_original_images.py
    
    # Extract to custom directory
    python scripts/extract_original_images.py --output data/original_only
    
    # Extract to subfolder in original directory
    python scripts/extract_original_images.py --output data/images/originals
        """
    )
    
    parser.add_argument('--images-dir', type=str, default='../data/images',
                        help='Source images directory (default: data/images)')
    parser.add_argument('--output', '-o', type=str, default='../data/images_original',
                        help='Output directory (default: data/images_original)')
    parser.add_argument('--max-number', type=int, default=101,
                        help='Maximum FloorPlan number (default: 101)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview mode, do not actually copy files')
    
    args = parser.parse_args()
    
    # Check source directory
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory does not exist: {args.images_dir}")
        return
    
    # If output directory exists, ask for confirmation
    if os.path.exists(args.output) and not args.dry_run:
        files_in_output = len(os.listdir(args.output))
        if files_in_output > 0:
            print(f"Warning: Output directory exists and contains {files_in_output} files")
            print(f"      {args.output}")
            response = input("Continuing will add files to this directory. Continue? (y/n): ")
            if response.lower() not in ['y', 'yes']:
                print("Cancelled")
                return
            print()
    
    # Execute extraction
    extracted, skipped = extract_original_images(
        args.images_dir,
        args.output,
        args.max_number,
        args.dry_run
    )


if __name__ == '__main__':
    main()
