"""
Floor Plan Analysis Agent
Integrate furniture detection, room segmentation, area calculation and all other features
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import json
from detection.detect_furniture import FurnitureDetector
from segmentation.segment_room import RoomSegmenter
from utils.area_calculator import AreaCalculator

class FloorPlanAgent:
    """Floor Plan Intelligent Analysis Agent"""
    
    def __init__(self, 
                 detection_model_path='runs/detect/furniture_detection/weights/best.pt',
                 segmentation_model_path='models/segmentation/best_model.pth',
                 segmentation_config='config/room_segmentation.yaml'):
        """
        Initialize Agent
        
        Args:
            detection_model_path: Furniture detection model path
            segmentation_model_path: Room segmentation model path
            segmentation_config: Segmentation configuration file path
        """
        print("Initializing Floor Plan Analysis Agent...")
        
        # Initialize components
        try:
            self.furniture_detector = FurnitureDetector(detection_model_path)
            print("[OK] Furniture detector loaded successfully")
        except Exception as e:
            print(f"[ERROR] Furniture detector loading failed: {e}")
            self.furniture_detector = None
        
        try:
            self.room_segmenter = RoomSegmenter(segmentation_model_path, segmentation_config)
            print("[OK] Room segmenter loaded successfully")
        except Exception as e:
            print(f"[ERROR] Room segmenter loading failed: {e}")
            self.room_segmenter = None
        
        self.area_calculator = AreaCalculator()
        print("[OK] Area calculator initialized successfully")
        
        print("\nAgent initialization complete!\n")
    
    def analyze(self, image_path, scale=None, reference_length=None, 
                reference_pixels=None, unit='m2', save_dir=None):
        """
        Complete analysis of a floor plan
        
        Args:
            image_path: Floor plan path
            scale: Scale string, e.g., "1:100"
            reference_length: Reference length (cm)
            reference_pixels: Reference pixel length
            unit: Area unit
            save_dir: Directory to save results
            
        Returns:
            dict: Complete analysis results
        """
        print(f"\n{'='*60}")
        print(f"Starting floor plan analysis: {image_path}")
        print(f"{'='*60}\n")
        
        results = {
            'image_path': str(image_path),
            'furniture': None,
            'rooms': None,
            'area': None,
            'success': False
        }
        
        # 1. Furniture Detection
        if self.furniture_detector is not None:
            print("[Step 1/3] Furniture Symbol Detection")
            print("-" * 60)
            try:
                det_results, furniture_stats = self.furniture_detector.detect(image_path)
                results['furniture'] = furniture_stats
                self.furniture_detector.print_statistics(furniture_stats)
                
                # Save visualization
                if save_dir:
                    save_path = Path(save_dir) / f"{Path(image_path).stem}_furniture.jpg"
                    self.furniture_detector.visualize(image_path, det_results, str(save_path))
                
                print("[OK] Furniture detection complete\n")
            except Exception as e:
                print(f"[ERROR] Furniture detection failed: {e}\n")
        else:
            print("[Step 1/3] Skipped (detector not loaded)\n")
        
        # 2. Room Segmentation
        if self.room_segmenter is not None:
            print("[Step 2/3] Room Semantic Segmentation")
            print("-" * 60)
            try:
                mask, colored_mask = self.room_segmenter.segment(image_path)
                rooms = self.room_segmenter.extract_rooms(mask)
                results['rooms'] = rooms
                
                print(f"[OK] Detected {len(rooms)} rooms")
                
                # Save visualization
                if save_dir:
                    save_path = Path(save_dir) / f"{Path(image_path).stem}_segmentation.jpg"
                    self.room_segmenter.visualize(image_path, mask, colored_mask, str(save_path))
                
                print("[OK] Room segmentation complete\n")
            except Exception as e:
                print(f"[ERROR] Room segmentation failed: {e}\n")
                rooms = []
        else:
            print("[Step 2/3] Skipped (segmenter not loaded)\n")
            rooms = []
        
        # 3. Area Calculation
        print("[Step 3/3] Area Calculation")
        print("-" * 60)
        
        try:
            # Set scale
            if scale:
                self.area_calculator.set_scale_manual(scale)
            else:
                # Try auto-detection
                if not self.area_calculator.detect_scale_from_image(image_path):
                    print("Warning: Could not detect scale, please set manually")
            
            # Calibration
            if reference_pixels and reference_length:
                self.area_calculator.calibrate_with_reference(reference_pixels, reference_length)
            elif self.area_calculator.scale_ratio:
                # If only have scale, prompt user for calibration
                print("Tip: For accurate area calculation, calibration using reference line is recommended")
                print("      Please provide --reference-pixels and --reference-length parameters")
            
            # Calculate area
            if rooms and (reference_pixels and reference_length):
                area_results = self.area_calculator.calculate_rooms_area(rooms, unit)
                results['area'] = area_results
                self.area_calculator.print_area_report(area_results)
                print("[OK] Area calculation complete\n")
            else:
                print("[WARN] Room data and calibration information required for area calculation\n")
                
        except Exception as e:
            print(f"[ERROR] Area calculation failed: {e}\n")
        
        # Mark success
        results['success'] = True
        
        print(f"{'='*60}")
        print("Analysis Complete!")
        print(f"{'='*60}\n")
        
        return results
    
    def save_results(self, results, output_path):
        """
        Save analysis results as JSON
        
        Args:
            results: Analysis results
            output_path: Output path
        """
        # Convert numpy types to Python native types
        def convert_types(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_clean = convert_types(results)
        
        # Remove large objects that cannot be serialized
        if results_clean.get('rooms'):
            for room in results_clean['rooms']:
                if 'mask' in room:
                    del room['mask']  # mask is too large, don't save
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_clean, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {output_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Floor Plan Intelligent Analysis Agent')
    parser.add_argument('--image', type=str, required=True,
                       help='Input floor plan path')
    parser.add_argument('--detection-model', type=str,
                       default='runs/detect/furniture_detection/weights/best.pt',
                       help='Furniture detection model path')
    parser.add_argument('--segmentation-model', type=str,
                       default='models/segmentation/best_model.pth',
                       help='Room segmentation model path')
    parser.add_argument('--scale', type=str, default=None,
                       help='Scale, e.g., "1:100"')
    parser.add_argument('--reference-pixels', type=float, default=None,
                       help='Pixel length of reference line')
    parser.add_argument('--reference-length', type=float, default=None,
                       help='Actual length of reference line (cm)')
    parser.add_argument('--unit', type=str, default='m2',
                       choices=['m2', 'cm2', 'ft2'],
                       help='Area unit')
    parser.add_argument('--save-dir', type=str, default='output',
                       help='Directory to save results')
    parser.add_argument('--json', type=str, default=None,
                       help='Path to save JSON results')
    
    args = parser.parse_args()
    
    # Create save directory
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create Agent
    agent = FloorPlanAgent(
        detection_model_path=args.detection_model,
        segmentation_model_path=args.segmentation_model
    )
    
    # Execute analysis
    results = agent.analyze(
        image_path=args.image,
        scale=args.scale,
        reference_length=args.reference_length,
        reference_pixels=args.reference_pixels,
        unit=args.unit,
        save_dir=args.save_dir
    )
    
    # Save JSON results
    if args.json:
        agent.save_results(results, args.json)
    elif args.save_dir:
        json_path = Path(args.save_dir) / f"{Path(args.image).stem}_results.json"
        agent.save_results(results, str(json_path))

if __name__ == '__main__':
    main()
