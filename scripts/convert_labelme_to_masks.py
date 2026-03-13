#!/usr/bin/env python3
"""
Convert LabelMe JSON annotations to segmentation masks
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import argparse


def labelme_to_mask(json_path, class_mapping):
    """
    Convert a single LabelMe JSON file to segmentation mask
    
    Args:
        json_path: Path to JSON file
        class_mapping: Dictionary mapping class names to IDs
        
    Returns:
        mask: Segmentation mask (H x W)
    """
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get image dimensions
    height = data['imageHeight']
    width = data['imageWidth']
    
    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Process each shape
    for shape in data['shapes']:
        label = shape['label'].lower()
        points = shape['points']
        
        # Skip unknown labels
        if label not in class_mapping:
            print(f"  Warning: Unknown label '{label}', skipping")
            continue
        
        class_id = class_mapping[label]
        
        # Convert points to numpy array
        points_np = np.array(points, dtype=np.int32)
        
        # Fill polygon
        cv2.fillPoly(mask, [points_np], color=class_id)
    
    return mask


def visualize_mask(mask, class_mapping):
    """
    Create colored visualization of segmentation mask
    
    Args:
        mask: Segmentation mask
        class_mapping: Class mapping dictionary
        
    Returns:
        colored_mask: RGB visualization
    """
    # Define colors for each class
    colors = {
        0: [0, 0, 0],         # background - black
        1: [128, 128, 128],   # wall - gray
        2: [0, 255, 0]        # room - green
    }
    
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in colors.items():
        colored[mask == class_id] = color
    
    return colored


def convert_directory(json_dir, output_mask_dir, output_viz_dir=None):
    """
    Convert all JSON files in a directory
    
    Args:
        json_dir: Directory containing JSON files
        output_mask_dir: Directory to save masks
        output_viz_dir: Directory to save visualizations (optional)
    """
    # Class mapping
    class_mapping = {
        'background': 0,
        'wall': 1,
        'room': 2
    }
    
    json_dir = Path(json_dir)
    output_mask_dir = Path(output_mask_dir)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    
    if output_viz_dir:
        output_viz_dir = Path(output_viz_dir)
        output_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files
    json_files = list(json_dir.glob('*.json'))
    
    if len(json_files) == 0:
        print(f"\nError: No JSON files found in {json_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"Converting LabelMe JSON to Segmentation Masks")
    print(f"{'='*80}\n")
    print(f"Input directory:  {json_dir}")
    print(f"Output directory: {output_mask_dir}")
    if output_viz_dir:
        print(f"Visualization:    {output_viz_dir}")
    print(f"\nFound {len(json_files)} JSON files\n")
    
    success_count = 0
    
    for json_path in json_files:
        print(f"Converting: {json_path.name}")
        
        try:
            # Convert to mask
            mask = labelme_to_mask(json_path, class_mapping)
            
            # Save mask
            stem = json_path.stem
            mask_path = output_mask_dir / f"{stem}.png"
            cv2.imwrite(str(mask_path), mask)
            
            # Print statistics
            unique_classes = np.unique(mask)
            class_names = [k for k, v in class_mapping.items() if v in unique_classes]
            print(f"  Classes found: {', '.join(class_names)}")
            print(f"  Saved mask: {mask_path.name}")
            
            # Save visualization if requested
            if output_viz_dir:
                colored = visualize_mask(mask, class_mapping)
                viz_path = output_viz_dir / f"{stem}_viz.png"
                cv2.imwrite(str(viz_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
                print(f"  Saved viz:  {viz_path.name}")
            
            success_count += 1
            print()
            
        except Exception as e:
            print(f"  Error: {e}\n")
    
    print(f"{'='*80}")
    print(f"Conversion Complete")
    print(f"{'='*80}")
    print(f"Successfully converted: {success_count}/{len(json_files)} files")
    print(f"Masks saved to: {output_mask_dir}")
    if output_viz_dir:
        print(f"Visualizations saved to: {output_viz_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Convert LabelMe JSON annotations to segmentation masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python scripts/convert_labelme_to_masks.py \\
        --json-dir data/labels_segmentation/json \\
        --output data/labels_segmentation/masks
    
    # With visualization
    python scripts/convert_labelme_to_masks.py \\
        --json-dir data/labels_segmentation/json \\
        --output data/labels_segmentation/masks \\
        --viz data/labels_segmentation/visualizations
        """
    )
    
    parser.add_argument('--json-dir', type=str, required=True,
                        help='Directory containing LabelMe JSON files')
    parser.add_argument('--output', type=str, required=True,
                        help='Directory to save segmentation masks')
    parser.add_argument('--viz', type=str,
                        help='Directory to save colored visualizations (optional)')
    
    args = parser.parse_args()
    
    # Check input directory
    if not os.path.exists(args.json_dir):
        print(f"Error: JSON directory not found: {args.json_dir}")
        return
    
    # Convert
    convert_directory(args.json_dir, args.output, args.viz)


if __name__ == '__main__':
    main()

