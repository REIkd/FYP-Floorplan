#!/usr/bin/env python3
"""
Data Augmentation for Segmentation Dataset
Generate multiple augmented versions from each annotated image
"""

import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
import argparse
from tqdm import tqdm


def create_augmentation_pipeline():
    """Create comprehensive augmentation pipeline"""
    return A.Compose([
        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Rotate(limit=90, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Rotate(limit=180, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Rotate(limit=270, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0),
        ], p=1.0),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.5),
    ])


def augment_single_image(image, mask, num_augmentations=10):
    """
    Generate multiple augmented versions of a single image
    
    Args:
        image: Input image (numpy array)
        mask: Segmentation mask (numpy array)
        num_augmentations: Number of augmented versions to generate
        
    Returns:
        List of (augmented_image, augmented_mask) tuples
    """
    augmentation = create_augmentation_pipeline()
    augmented_pairs = []
    
    for i in range(num_augmentations):
        augmented = augmentation(image=image, mask=mask)
        augmented_pairs.append((augmented['image'], augmented['mask']))
    
    return augmented_pairs


def augment_dataset(images_dir, masks_dir, output_images_dir, output_masks_dir, 
                   num_augmentations=10):
    """
    Augment entire dataset
    
    Args:
        images_dir: Directory containing original images
        masks_dir: Directory containing original masks
        output_images_dir: Directory to save augmented images
        output_masks_dir: Directory to save augmented masks
        num_augmentations: Number of augmented versions per image
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_images_dir = Path(output_images_dir)
    output_masks_dir = Path(output_masks_dir)
    
    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.JPG'))
    
    # Filter images that have corresponding masks
    valid_images = []
    for img_path in image_files:
        mask_name = img_path.stem + '.png'
        mask_path = masks_dir / mask_name
        if mask_path.exists():
            valid_images.append(img_path)
    
    print("=" * 80)
    print("Data Augmentation for Segmentation")
    print("=" * 80)
    print(f"Original images: {len(valid_images)}")
    print(f"Augmentations per image: {num_augmentations}")
    print(f"Total images after augmentation: {len(valid_images) * (num_augmentations + 1)}")
    print()
    print(f"Output images: {output_images_dir}")
    print(f"Output masks: {output_masks_dir}")
    print()
    
    total_generated = 0
    
    # Process each image
    for img_path in tqdm(valid_images, desc="Augmenting images"):
        # Read image and mask
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_name = img_path.stem + '.png'
        mask_path = masks_dir / mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Copy original
        original_img_out = output_images_dir / img_path.name
        original_mask_out = output_masks_dir / mask_name
        cv2.imwrite(str(original_img_out), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(original_mask_out), mask)
        total_generated += 1
        
        # Generate augmented versions
        augmented_pairs = augment_single_image(image, mask, num_augmentations)
        
        for i, (aug_img, aug_mask) in enumerate(augmented_pairs, 1):
            # Generate output filenames
            aug_img_name = f"{img_path.stem}_aug{i:02d}{img_path.suffix}"
            aug_mask_name = f"{img_path.stem}_aug{i:02d}.png"
            
            aug_img_path = output_images_dir / aug_img_name
            aug_mask_path = output_masks_dir / aug_mask_name
            
            # Save
            cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(aug_mask_path), aug_mask)
            total_generated += 1
    
    print()
    print("=" * 80)
    print("Augmentation Complete!")
    print("=" * 80)
    print(f"Total images generated: {total_generated}")
    print(f"  Original: {len(valid_images)}")
    print(f"  Augmented: {total_generated - len(valid_images)}")
    print()
    print(f"Images saved to: {output_images_dir}")
    print(f"Masks saved to: {output_masks_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Augment segmentation dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # Generate 10 augmented versions per image
    python scripts/augment_segmentation_data.py \\
        --images data/images_original \\
        --masks data/labels_segmentation/masks \\
        --output-images data/segmentation_augmented/images \\
        --output-masks data/segmentation_augmented/masks \\
        --num-aug 10
        """
    )
    
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing original images')
    parser.add_argument('--masks', type=str, required=True,
                        help='Directory containing original masks')
    parser.add_argument('--output-images', type=str, required=True,
                        help='Directory to save augmented images')
    parser.add_argument('--output-masks', type=str, required=True,
                        help='Directory to save augmented masks')
    parser.add_argument('--num-aug', type=int, default=10,
                        help='Number of augmentations per image (default: 10)')
    
    args = parser.parse_args()
    
    augment_dataset(
        args.images,
        args.masks,
        args.output_images,
        args.output_masks,
        args.num_aug
    )


if __name__ == '__main__':
    main()

