#!/usr/bin/env python3
"""
Complete Room Area Calculation Pipeline
Combines object detection and room segmentation to calculate room areas
"""

import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from ultralytics import YOLO
from pathlib import Path
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AreaCalculator:
    """Calculate actual areas from pixel measurements"""
    
    def __init__(self):
        self.pixels_per_cm = None
        self.scale_ratio = None
    
    def calibrate_with_reference(self, pixel_length, actual_length_cm):
        """
        Calibrate using a reference measurement
        
        Args:
            pixel_length: Length in pixels
            actual_length_cm: Actual length in centimeters
        """
        self.pixels_per_cm = pixel_length / actual_length_cm
        print(f"Calibration: {pixel_length} pixels = {actual_length_cm} cm")
        print(f"Ratio: {self.pixels_per_cm:.4f} pixels/cm")
    
    def set_scale(self, scale_ratio):
        """Set scale ratio (e.g., 100 for 1:100)"""
        self.scale_ratio = scale_ratio
        print(f"Scale set to 1:{scale_ratio}")
    
    def pixels_to_area(self, area_pixels, unit='m2'):
        """Convert pixel area to actual area"""
        if self.pixels_per_cm is None:
            raise ValueError("Please calibrate first!")
        
        # Convert to cm²
        area_cm2 = area_pixels / (self.pixels_per_cm ** 2)
        
        if unit == 'm2':
            return area_cm2 / 10000
        elif unit == 'cm2':
            return area_cm2
        elif unit == 'ft2':
            return area_cm2 * 0.00107639
        else:
            raise ValueError(f"Unsupported unit: {unit}")


class RoomAreaAnalyzer:
    """Complete room area analysis"""
    
    def __init__(self, detection_model_path, segmentation_model_path):
        """Initialize analyzer with both models"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load detection model (YOLOv8)
        print("Loading furniture detection model...")
        self.detector = YOLO(detection_model_path)
        print("  [OK] Detection model loaded")
        
        # Load segmentation model (U-Net)
        print("Loading room segmentation model...")
        num_classes = 5
        self.segmenter = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,
            classes=num_classes,
            activation=None
        )
        self.segmenter.load_state_dict(
            torch.load(segmentation_model_path, map_location=self.device)
        )
        self.segmenter = self.segmenter.to(self.device)
        self.segmenter.eval()
        print("  [OK] Segmentation model loaded")
        
        # Transform for segmentation
        self.seg_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Area calculator
        self.calculator = AreaCalculator()
    
    def segment_image(self, image_path):
        """Run segmentation on image"""
        # Read image
        image = cv2.imread(str(image_path))
        original_h, original_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform and predict
        augmented = self.seg_transform(image=image_rgb)
        image_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.segmenter(image_tensor)
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Resize to original
        mask = cv2.resize(mask.astype(np.uint8), (original_w, original_h),
                         interpolation=cv2.INTER_NEAREST)
        
        return mask, image
    
    def extract_rooms(self, mask):
        """Extract individual rooms from mask"""
        room_mask = (mask == 2).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            room_mask, connectivity=8
        )
        
        rooms = []
        for i in range(1, num_labels):
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
        
        rooms.sort(key=lambda r: r['area_pixels'], reverse=True)
        return rooms
    
    def detect_furniture(self, image_path):
        """Run furniture detection"""
        results = self.detector.predict(image_path, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy()
                
                detections.append({
                    'class': self.detector.names[cls_id],
                    'confidence': conf,
                    'bbox': bbox
                })
        
        return detections
    
    def analyze(self, image_path, pixel_length, actual_length_cm, unit='m2'):
        """
        Complete analysis pipeline
        
        Args:
            image_path: Path to floor plan image
            pixel_length: Reference line length in pixels
            actual_length_cm: Reference line actual length in cm
            unit: Output unit for areas
            
        Returns:
            Complete analysis results
        """
        print("\n" + "=" * 80)
        print("Floor Plan Analysis - Room Area Calculation")
        print("=" * 80)
        print(f"Image: {image_path}")
        print()
        
        # Calibrate
        print("[Step 1/4] Calibrating scale...")
        self.calculator.calibrate_with_reference(pixel_length, actual_length_cm)
        print()
        
        # Segment rooms
        print("[Step 2/4] Segmenting rooms...")
        mask, image = self.segment_image(image_path)
        rooms = self.extract_rooms(mask)
        print(f"  Detected {len(rooms)} rooms")
        print()
        
        # Calculate areas
        print("[Step 3/4] Calculating room areas...")
        for room in rooms:
            room['area_actual'] = self.calculator.pixels_to_area(
                room['area_pixels'], unit
            )
        total_area = sum(r['area_actual'] for r in rooms)
        print(f"  Total area: {total_area:.2f} {unit}")
        print()
        
        # Detect furniture
        print("[Step 4/4] Detecting furniture...")
        furniture = self.detect_furniture(image_path)
        print(f"  Detected {len(furniture)} furniture items")
        print()
        
        # Compile results
        results = {
            'image_path': str(image_path),
            'num_rooms': len(rooms),
            'rooms': rooms,
            'total_area': total_area,
            'unit': unit,
            'furniture': furniture,
            'mask': mask,
            'image': image
        }
        
        return results
    
    def print_report(self, results):
        """Print analysis report"""
        print("\n" + "=" * 80)
        print("FLOOR PLAN ANALYSIS REPORT")
        print("=" * 80)
        print()
        
        # Room areas
        print("ROOM AREAS:")
        print("-" * 80)
        unit = results['unit']
        for i, room in enumerate(results['rooms'], 1):
            print(f"  Room {i:2d}: {room['area_actual']:8.2f} {unit}  "
                  f"({room['area_pixels']:,} pixels)")
        print("-" * 80)
        print(f"  TOTAL:    {results['total_area']:8.2f} {unit}")
        print()
        
        # Furniture count
        print("FURNITURE SUMMARY:")
        print("-" * 80)
        furniture_count = {}
        for item in results['furniture']:
            cls = item['class']
            furniture_count[cls] = furniture_count.get(cls, 0) + 1
        
        for cls, count in sorted(furniture_count.items()):
            print(f"  {cls:20s}: {count:3d}")
        print("-" * 80)
        print(f"  Total items: {len(results['furniture'])}")
        print()
        
        print("=" * 80)
    
    def save_visualization(self, results, output_path):
        """Save visualization of results"""
        image = results['image']
        mask = results['mask']
        rooms = results['rooms']
        
        # Create colored mask
        colors = {
            0: [0, 0, 0],
            1: [128, 128, 128],
            2: [0, 255, 0],
            3: [255, 0, 0],
            4: [0, 0, 255]
        }
        
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            colored_mask[mask == class_id] = color
        
        # Create overlay
        overlay = cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)
        
        # Draw room labels
        unit = results['unit']
        for i, room in enumerate(rooms, 1):
            cx, cy = room['centroid']
            area = room['area_actual']
            
            # Draw circle
            cv2.circle(overlay, (cx, cy), 8, (255, 255, 0), -1)
            
            # Draw text background
            text = f"Room {i}: {area:.1f} {unit}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (cx - text_w//2 - 5, cy - 40),
                         (cx + text_w//2 + 5, cy - 15), (0, 0, 0), -1)
            
            # Draw text (replace m² with sqm for display)
            text_display = text.replace('m²', 'sqm').replace('m2', 'sqm')
            cv2.putText(overlay, text_display, (cx - text_w//2, cy - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save
        cv2.imwrite(output_path, overlay)
        print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate room areas from floor plan',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/calculate_room_areas.py \\
        --image data/images_original/FloorPlan-1.jpg \\
        --detection-model runs/detect/train_90/weights/best.pt \\
        --segmentation-model models/segmentation/quick_test_model.pth \\
        --ref-pixels 200 \\
        --ref-length 200 \\
        --unit m2 \\
        --output results/room_areas.jpg
        """
    )
    
    parser.add_argument('--image', type=str, required=True,
                        help='Path to floor plan image')
    parser.add_argument('--detection-model', type=str, 
                        default='runs/detect/train_90/weights/best.pt',
                        help='Path to furniture detection model')
    parser.add_argument('--segmentation-model', type=str,
                        default='models/segmentation/quick_test_model.pth',
                        help='Path to room segmentation model')
    parser.add_argument('--ref-pixels', type=float, required=True,
                        help='Reference line length in pixels')
    parser.add_argument('--ref-length', type=float, required=True,
                        help='Reference line actual length in cm')
    parser.add_argument('--unit', type=str, default='m2',
                        choices=['m2', 'cm2', 'ft2'],
                        help='Area unit (default: m2)')
    parser.add_argument('--output', type=str, default='results/room_areas.jpg',
                        help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RoomAreaAnalyzer(args.detection_model, args.segmentation_model)
    
    # Run analysis
    results = analyzer.analyze(
        args.image,
        args.ref_pixels,
        args.ref_length,
        args.unit
    )
    
    # Print report
    analyzer.print_report(results)
    
    # Save visualization
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer.save_visualization(results, args.output)
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()

