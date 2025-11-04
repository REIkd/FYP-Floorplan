"""
Dataset Visualization Tool
View annotations and check data quality
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import random

def visualize_yolo_annotations(images_dir, labels_dir, class_names, num_samples=10):
    """
    Visualize YOLO format annotations
    
    Args:
        images_dir: Images directory
        labels_dir: Labels directory
        class_names: List of class names
        num_samples: Number of samples to display
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # Get all images
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    # Randomly select samples
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"Displaying {len(samples)} annotation samples (press any key to continue, ESC to exit)")
    
    for img_path in samples:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Read labels
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            print(f"Warning: {img_path.name} has no corresponding label file")
            continue
        
        # Parse and draw annotations
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Convert to pixel coordinates
                x_center *= w
                y_center *= h
                width *= w
                height *= h
                
                # Calculate bounding box
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Draw
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Label text
                if class_id < len(class_names):
                    label = class_names[class_id]
                else:
                    label = f"Class {class_id}"
                
                cv2.putText(image, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display
        display = cv2.resize(image, (1200, 800))
        cv2.imshow('Annotation Visualization', display)
        
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()

def visualize_segmentation_masks(images_dir, masks_dir, num_samples=10):
    """
    Visualize segmentation masks
    
    Args:
        images_dir: Images directory
        masks_dir: Masks directory
        num_samples: Number of samples to display
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    # Get all images
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    # Randomly select samples
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Define color mapping
    colors = {
        0: [0, 0, 0],         # background
        1: [128, 128, 128],   # wall
        2: [255, 200, 200],   # room
        3: [100, 200, 100],   # door_area
        4: [100, 100, 200],   # window_area
    }
    
    print(f"Displaying {len(samples)} segmentation annotation samples (press any key to continue, ESC to exit)")
    
    for img_path in samples:
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Read mask
        mask_path = masks_dir / f"{img_path.stem}.png"
        
        if not mask_path.exists():
            print(f"Warning: {img_path.name} has no corresponding mask file")
            continue
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            colored_mask[mask == class_id] = color
        
        # Overlay display
        overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
        
        # Side-by-side display
        combined = np.hstack([image, colored_mask, overlay])
        
        # Resize
        display = cv2.resize(combined, (1800, 600))
        
        cv2.imshow('Segmentation Annotation Visualization (Original | Mask | Overlay)', display)
        
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break
    
    cv2.destroyAllWindows()

def analyze_dataset_statistics(images_dir, labels_dir, class_names):
    """
    Analyze dataset statistics
    
    Args:
        images_dir: Images directory
        labels_dir: Labels directory
        class_names: List of class names
    """
    from collections import Counter
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # Get all label files
    label_files = list(labels_dir.glob('*.txt'))
    
    if not label_files:
        print(f"No label files found in {labels_dir}")
        return
    
    # Statistics
    class_counts = Counter()
    total_objects = 0
    images_with_labels = 0
    
    for label_path in label_files:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if lines:
                images_with_labels += 1
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_objects += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Total images: {len(label_files)}")
    print(f"Images with labels: {images_with_labels}")
    print(f"Total objects: {total_objects}")
    print(f"Average per image: {total_objects/images_with_labels:.1f} objects")
    
    print("\nCount by class:")
    print("-"*60)
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        percentage = count / total_objects * 100
        print(f"  {class_name:20s}: {count:5d} ({percentage:5.1f}%)")
    print("="*60 + "\n")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize dataset annotations')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['detection', 'segmentation', 'stats'],
                       help='Visualization mode')
    parser.add_argument('--images-dir', type=str, default='data/images',
                       help='Images directory')
    parser.add_argument('--labels-dir', type=str, default=None,
                       help='Labels directory')
    parser.add_argument('--masks-dir', type=str, default=None,
                       help='Masks directory (for segmentation)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to display')
    
    args = parser.parse_args()
    
    # Default class names
    detection_classes = [
        'door', 'window', 'table', 'chair', 'bed', 'sofa',
        'toilet', 'sink', 'bathtub', 'stove', 'refrigerator',
        'wardrobe', 'tv', 'desk', 'washingmachine', 'loadbearing_wall',
        'aircondition', 'cupboard'
    ]
    
    if args.mode == 'detection':
        labels_dir = args.labels_dir or 'data/labels_detection'
        visualize_yolo_annotations(
            args.images_dir,
            labels_dir,
            detection_classes,
            args.num_samples
        )
    
    elif args.mode == 'segmentation':
        masks_dir = args.masks_dir or 'data/labels_segmentation/masks'
        visualize_segmentation_masks(
            args.images_dir,
            masks_dir,
            args.num_samples
        )
    
    elif args.mode == 'stats':
        labels_dir = args.labels_dir or 'data/labels_detection'
        analyze_dataset_statistics(
            args.images_dir,
            labels_dir,
            detection_classes
        )

if __name__ == '__main__':
    main()
