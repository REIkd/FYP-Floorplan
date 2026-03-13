#!/usr/bin/env python3
"""
Test trained segmentation model on new images
"""

import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RoomSegmenter:
    """Room Segmentation Inference"""
    
    def __init__(self, model_path, img_size=256):
        """
        Initialize segmenter
        
        Args:
            model_path: Path to trained model
            img_size: Image size for inference
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        
        # Load model
        num_classes = 5
        self.model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            classes=num_classes,
            activation=None
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transform
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Class names and colors
        self.class_names = {
            0: 'background',
            1: 'wall',
            2: 'room'
        }
        
        self.colors = {
            0: [0, 0, 0],         # background - black
            1: [128, 128, 128],   # wall - gray
            2: [0, 255, 0]        # room - green
        }
    
    def predict(self, image_path):
        """
        Predict segmentation mask for an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            mask: Predicted segmentation mask
            colored_mask: Colored visualization
            original_image: Original image
        """
        # Read image
        image = cv2.imread(str(image_path))
        original_image = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        
        # Transform
        augmented = self.transform(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Resize back to original size
        mask = cv2.resize(mask.astype(np.uint8), (original_w, original_h), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Create colored visualization
        colored_mask = self.create_colored_mask(mask)
        
        return mask, colored_mask, original_image
    
    def create_colored_mask(self, mask):
        """Create colored visualization of mask"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.colors.items():
            colored[mask == class_id] = color
        
        return colored
    
    def analyze_mask(self, mask):
        """Analyze mask and extract statistics"""
        stats = {}
        
        for class_id, class_name in self.class_names.items():
            area_pixels = np.sum(mask == class_id)
            percentage = (area_pixels / mask.size) * 100
            stats[class_name] = {
                'pixels': int(area_pixels),
                'percentage': round(percentage, 2)
            }
        
        return stats
    
    def extract_rooms(self, mask):
        """
        Extract individual room regions
        
        Returns:
            List of room dictionaries with area and bounding box
        """
        # Get room mask (class_id = 2)
        room_mask = (mask == 2).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            room_mask, connectivity=8
        )
        
        rooms = []
        for i in range(1, num_labels):  # Skip background (0)
            area_pixels = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            rooms.append({
                'id': i,
                'area_pixels': int(area_pixels),
                'bbox': (x, y, w, h),
                'centroid': (int(centroids[i][0]), int(centroids[i][1]))
            })
        
        # Sort by area
        rooms.sort(key=lambda r: r['area_pixels'], reverse=True)
        
        return rooms


def visualize_results(original_image, colored_mask, stats, rooms, output_path):
    """Create comprehensive visualization"""
    h, w = original_image.shape[:2]
    
    # Create overlay
    overlay = cv2.addWeighted(original_image, 0.6, colored_mask, 0.4, 0)
    
    # Draw room info
    for i, room in enumerate(rooms, 1):
        cx, cy = room['centroid']
        cv2.circle(overlay, (cx, cy), 5, (255, 255, 0), -1)
        cv2.putText(overlay, f"Room {i}", (cx - 30, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(overlay, f"{room['area_pixels']} px", (cx - 40, cy + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Create info panel
    info_panel = np.ones((200, w, 3), dtype=np.uint8) * 255
    
    y_offset = 30
    cv2.putText(info_panel, "Segmentation Statistics:", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    y_offset += 30
    for class_name, data in stats.items():
        if data['percentage'] > 0.1:  # Only show classes with >0.1%
            text = f"{class_name}: {data['percentage']:.1f}%"
            cv2.putText(info_panel, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25
    
    y_offset += 10
    cv2.putText(info_panel, f"Total Rooms Detected: {len(rooms)}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Combine all
    result = np.vstack([overlay, info_panel])
    
    # Save
    cv2.imwrite(output_path, result)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Test segmentation model')
    parser.add_argument('--model', type=str, default='models/segmentation/quick_test_model.pth',
                        help='Path to trained model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, default='results/segmentation_test.jpg',
                        help='Path to save result')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Image size for inference')
    
    args = parser.parse_args()
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return
    
    # Check image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    print("=" * 80)
    print("Room Segmentation Test")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print()
    
    # Load segmenter
    print("Loading model...")
    segmenter = RoomSegmenter(args.model, args.img_size)
    print("Model loaded successfully!")
    print()
    
    # Predict
    print("Running segmentation...")
    mask, colored_mask, original_image = segmenter.predict(args.image)
    print("Segmentation complete!")
    print()
    
    # Analyze
    print("Analyzing results...")
    stats = segmenter.analyze_mask(mask)
    rooms = segmenter.extract_rooms(mask)
    
    print("=" * 80)
    print("Segmentation Results")
    print("=" * 80)
    print()
    print("Area Statistics:")
    for class_name, data in stats.items():
        if data['percentage'] > 0.1:
            print(f"  {class_name:15s}: {data['percentage']:6.2f}% ({data['pixels']:,} pixels)")
    print()
    
    print(f"Detected Rooms: {len(rooms)}")
    for i, room in enumerate(rooms, 1):
        print(f"  Room {i}: {room['area_pixels']:,} pixels at {room['centroid']}")
    print()
    
    # Visualize
    print("Creating visualization...")
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_results(original_image, colored_mask, stats, rooms, args.output)
    print(f"Results saved to: {args.output}")
    print()
    
    print("=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

