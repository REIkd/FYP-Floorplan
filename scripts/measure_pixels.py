"""
Interactive Measurement Tool
Measure pixel distances in images to help calibrate scale
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

class PixelMeasureTool:
    """Pixel Measurement Tool"""
    
    def __init__(self, image_path):
        """Initialize"""
        self.image_path = image_path
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        self.display_image = self.image.copy()
        self.points = []
        self.measurements = []
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.points.append((x, y))
            
            # Draw point
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            
            if len(self.points) == 2:
                # Draw line
                cv2.line(self.display_image, self.points[0], self.points[1], 
                        (0, 255, 0), 2)
                
                # Calculate distance
                p1, p2 = self.points
                distance = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                
                # Display distance
                mid_point = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
                text = f"{distance:.1f} px"
                cv2.putText(self.display_image, text, mid_point,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                self.measurements.append({
                    'point1': p1,
                    'point2': p2,
                    'distance': distance
                })
                
                print(f"\nMeasurement #{len(self.measurements)}")
                print(f"  Start point: {p1}")
                print(f"  End point: {p2}")
                print(f"  Distance: {distance:.2f} pixels")
                
                # Reset points
                self.points = []
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: reset
            self.reset()
    
    def reset(self):
        """Reset image"""
        self.display_image = self.image.copy()
        self.points = []
        print("\nImage reset")
    
    def run(self):
        """Run measurement tool"""
        window_name = f'Pixel Measurement Tool - {Path(self.image_path).name}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("="*60)
        print("Pixel Measurement Tool")
        print("="*60)
        print("Instructions:")
        print("  - Left click: Select two points to measure distance")
        print("  - Right click: Reset image")
        print("  - Press 'r': Reset image")
        print("  - Press 's': Save annotated image")
        print("  - Press 'q' or ESC: Exit")
        print("="*60)
        
        while True:
            cv2.imshow(window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('r'):  # reset
                self.reset()
            elif key == ord('s'):  # save
                output_path = Path(self.image_path).stem + '_measured.jpg'
                cv2.imwrite(output_path, self.display_image)
                print(f"\nAnnotated image saved: {output_path}")
        
        cv2.destroyAllWindows()
        
        # Print summary
        if self.measurements:
            print("\n" + "="*60)
            print("Measurement Summary")
            print("="*60)
            for i, m in enumerate(self.measurements, 1):
                print(f"Measurement {i}: {m['distance']:.2f} pixels")
            
            avg_distance = np.mean([m['distance'] for m in self.measurements])
            print(f"\nAverage distance: {avg_distance:.2f} pixels")
            print("="*60)
            
            # Usage suggestions
            print("\nTips:")
            print(f"If these line segments all have actual length X cm, then:")
            print(f"  --reference-pixels {avg_distance:.0f}")
            print(f"  --reference-length X")
            print("\nFor example, if actual length is 200cm:")
            print(f"  --reference-pixels {avg_distance:.0f} --reference-length 200")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Measure pixel distances in image')
    parser.add_argument('--image', type=str, required=True,
                       help='Image path')
    
    args = parser.parse_args()
    
    try:
        tool = PixelMeasureTool(args.image)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
