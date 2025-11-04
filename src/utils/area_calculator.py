"""
Area Calculation Tool
Convert pixel area to actual area based on scale
"""

import re
import numpy as np
import cv2
import easyocr

class AreaCalculator:
    """Area Calculator"""
    
    def __init__(self):
        """Initialize calculator"""
        self.scale_ratio = None  # Scale ratio (e.g., 100 in 1:100)
        self.pixels_per_unit = None  # Pixels per unit length
        
    def set_scale_manual(self, scale_str):
        """
        Manually set scale
        
        Args:
            scale_str: Scale string, e.g., "1:100" or "1/100"
            
        Returns:
            bool: Whether setting succeeded
        """
        # Parse scale
        match = re.search(r'1[:\/](\d+)', scale_str)
        if match:
            self.scale_ratio = int(match.group(1))
            print(f"Scale set to: 1:{self.scale_ratio}")
            return True
        else:
            print(f"Cannot parse scale: {scale_str}")
            return False
    
    def detect_scale_from_image(self, image_path):
        """
        Auto-detect scale from image
        Use OCR to recognize scale text in image
        
        Args:
            image_path: Image path
            
        Returns:
            bool: Whether detection succeeded
        """
        try:
            # Initialize OCR
            reader = easyocr.Reader(['en', 'ch_sim'])
            
            # Read image
            image = cv2.imread(str(image_path))
            
            # OCR recognition
            results = reader.readtext(image)
            
            # Find scale in recognition results
            for (bbox, text, conf) in results:
                # Find scale format
                match = re.search(r'1[:\/](\d+)', text)
                if match:
                    self.scale_ratio = int(match.group(1))
                    print(f"Auto-detected scale: 1:{self.scale_ratio} (confidence: {conf:.2f})")
                    return True
            
            print("Could not detect scale from image")
            return False
            
        except Exception as e:
            print(f"OCR recognition failed: {e}")
            return False
    
    def calibrate_with_reference(self, pixel_length, actual_length_cm):
        """
        Calibrate using reference line segment
        
        Args:
            pixel_length: Pixel length of reference line
            actual_length_cm: Actual length of reference line (cm)
        """
        self.pixels_per_unit = pixel_length / actual_length_cm
        print(f"Calibration complete: {pixel_length} pixels = {actual_length_cm} cm")
        print(f"Pixels per cm: {self.pixels_per_unit:.2f}")
    
    def calculate_area_from_pixels(self, area_pixels, unit='m2'):
        """
        Convert pixel area to actual area
        
        Args:
            area_pixels: Pixel area
            unit: Output unit ('m2', 'cm2', 'ft2')
            
        Returns:
            float: Actual area
        """
        if self.scale_ratio is None and self.pixels_per_unit is None:
            raise ValueError("Please set scale or calibrate first!")
        
        if self.pixels_per_unit is not None:
            # Use calibration method
            area_cm2 = area_pixels / (self.pixels_per_unit ** 2)
        else:
            # Use scale method
            # Standard DPI assumption (simplified, may need user input in practice)
            # Typically floor plans have scale notation, 1:100 means 1cm on plan = 100cm actual
            # Here we assume standard print/scan resolution
            # User should provide a reference length for calibration
            raise ValueError("Using scale method requires calling calibrate_with_reference() first")
        
        # Convert units
        if unit == 'm2':
            return area_cm2 / 10000  # cm² to m²
        elif unit == 'cm2':
            return area_cm2
        elif unit == 'ft2':
            return area_cm2 * 0.00107639  # cm² to ft²
        else:
            raise ValueError(f"Unsupported unit: {unit}")
    
    def calculate_rooms_area(self, rooms, unit='m2'):
        """
        Calculate areas of multiple rooms
        
        Args:
            rooms: Room list (from segmenter)
            unit: Unit
            
        Returns:
            dict: Area of each room and total area
        """
        results = {
            'rooms': [],
            'total_area': 0,
            'unit': unit
        }
        
        for i, room in enumerate(rooms, 1):
            area = self.calculate_area_from_pixels(room['area_pixels'], unit)
            results['rooms'].append({
                'id': i,
                'area': area,
                'area_pixels': room['area_pixels']
            })
            results['total_area'] += area
        
        return results
    
    def print_area_report(self, area_results):
        """Print area report"""
        print("\n" + "="*50)
        print("Room Area Statistics")
        print("="*50)
        
        unit = area_results['unit']
        
        for room in area_results['rooms']:
            print(f"Room {room['id']:2d}: {room['area']:8.2f} {unit} "
                  f"({room['area_pixels']} pixels)")
        
        print("-"*50)
        print(f"Total area: {area_results['total_area']:8.2f} {unit}")
        print("="*50 + "\n")

def main():
    """Main function - Usage demo"""
    calculator = AreaCalculator()
    
    # Method 1: Manually set scale
    calculator.set_scale_manual("1:100")
    
    # Method 2: Calibrate using reference line
    # Assume user measured a line segment on the image
    # This line's pixel length is 200 pixels
    # According to scale, this line's actual length is 200cm (because 1:100)
    calculator.calibrate_with_reference(pixel_length=200, actual_length_cm=200)
    
    # Calculate area
    test_area_pixels = 50000  # Assume room area is 50000 pixels
    area_m2 = calculator.calculate_area_from_pixels(test_area_pixels, 'm2')
    print(f"Test: {test_area_pixels} pixels = {area_m2:.2f} m²")

if __name__ == '__main__':
    main()
