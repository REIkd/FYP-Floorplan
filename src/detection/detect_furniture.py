"""
Furniture Detection Inference Script
Use trained model to detect and count furniture in floor plans
"""

import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
import json

class FurnitureDetector:
    """Furniture Detector Class"""
    
    def __init__(self, model_path='runs/detect/furniture_detection/weights/best.pt'):
        """
        Initialize detector
        
        Args:
            model_path: Trained model path
        """
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
    def detect(self, image_path, conf_threshold=0.25):
        """
        Detect furniture in image
        
        Args:
            image_path: Image path
            conf_threshold: Confidence threshold
            
        Returns:
            results: Detection results
            statistics: Statistics information
        """
        # Execute detection
        results = self.model(image_path, conf=conf_threshold)[0]
        
        # Count furniture by class
        statistics = self._count_objects(results)
        
        return results, statistics
    
    def _count_objects(self, results):
        """Count detected objects"""
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        class_names = [self.class_names[i] for i in class_ids]
        
        # Count objects by class
        counter = Counter(class_names)
        
        statistics = {
            'total': len(class_ids),
            'by_class': dict(counter),
            'details': []
        }
        
        # Detailed information
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            xyxy = box.xyxy.cpu().numpy()[0]
            
            statistics['details'].append({
                'class': self.class_names[cls_id],
                'confidence': conf,
                'bbox': xyxy.tolist()
            })
        
        return statistics
    
    def visualize(self, image_path, results, save_path=None):
        """
        Visualize detection results
        
        Args:
            image_path: Original image path
            results: Detection results
            save_path: Save path
        """
        # Read original image
        img = cv2.imread(str(image_path))
        
        # Draw detection boxes
        annotated_img = results.plot()
        
        # Save or display
        if save_path:
            cv2.imwrite(save_path, annotated_img)
            print(f"Visualization result saved to: {save_path}")
        
        return annotated_img
    
    def print_statistics(self, statistics):
        """Print statistics"""
        print("\n" + "="*50)
        print("Furniture Detection Statistics")
        print("="*50)
        print(f"Total furniture detected: {statistics['total']}")
        print("\nCount by furniture type:")
        print("-"*50)
        
        for class_name, count in sorted(statistics['by_class'].items()):
            print(f"  {class_name:20s}: {count:3d} items")
        
        print("="*50 + "\n")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect furniture in floor plan')
    parser.add_argument('--image', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--model', type=str,
                       default='runs/detect/furniture_detection/weights/best.pt',
                       help='Model path')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--save', type=str, default=None,
                       help='Save path for visualization result')
    parser.add_argument('--json', type=str, default=None,
                       help='Save path for JSON results')
    
    args = parser.parse_args()
    
    # Create detector
    detector = FurnitureDetector(args.model)
    
    # Execute detection
    print(f"Detecting: {args.image}")
    results, statistics = detector.detect(args.image, args.conf)
    
    # Print statistics
    detector.print_statistics(statistics)
    
    # Save JSON results
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        print(f"Statistics saved to: {args.json}")
    
    # Visualize
    if args.save:
        detector.visualize(args.image, results, args.save)

if __name__ == '__main__':
    main()
