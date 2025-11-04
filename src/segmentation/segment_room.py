"""
Room Segmentation Inference Script
Use trained model for room and wall segmentation
"""

import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
import yaml

class RoomSegmenter:
    """Room Segmenter Class"""
    
    def __init__(self, model_path, config_path='config/room_segmentation.yaml'):
        """
        Initialize segmenter
        
        Args:
            model_path: Trained model path
            config_path: Configuration file path
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.class_names = list(self.config['classes'].values())
        
    def _load_model(self, model_path):
        """Load trained model"""
        model_cfg = self.config['model']
        num_classes = self.config['num_classes']
        
        # Create model structure
        if model_cfg['name'].lower() == 'unet':
            model = smp.Unet(
                encoder_name=model_cfg['encoder'],
                encoder_weights=None,  # Don't use pretrained weights
                classes=num_classes
            )
        # Add other models...
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def segment(self, image_path):
        """
        Segment image
        
        Args:
            image_path: Image path
            
        Returns:
            mask: Segmentation result mask
            colored_mask: Colored visualization mask
        """
        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # Preprocessing
        img_size = self.config['data']['img_size']
        image_resized = cv2.resize(image, (img_size[1], img_size[0]))
        image_normalized = image_resized / 255.0
        image_normalized = (image_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image_tensor = torch.from_numpy(image_normalized).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(image_tensor)
            mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Resize back to original size
        mask = cv2.resize(mask.astype(np.uint8), 
                         (original_shape[1], original_shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Create colored mask
        colored_mask = self._create_colored_mask(mask)
        
        return mask, colored_mask
    
    def _create_colored_mask(self, mask):
        """Create colored segmentation mask"""
        # Define color for each class
        colors = {
            0: [0, 0, 0],         # background - black
            1: [128, 128, 128],   # wall - gray
            2: [255, 200, 200],   # room - light red
            3: [100, 200, 100],   # door_area - green
            4: [100, 100, 200],   # window_area - blue
        }
        
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in colors.items():
            colored_mask[mask == class_id] = color
        
        return colored_mask
    
    def extract_rooms(self, mask):
        """
        Extract individual rooms
        
        Args:
            mask: Segmentation mask
            
        Returns:
            rooms: List of rooms, each room is a binary mask
        """
        # Extract room class mask
        room_mask = (mask == 2).astype(np.uint8) * 255
        
        # Morphological operations
        if self.config['postprocess'].get('morphology', True):
            kernel = np.ones((5, 5), np.uint8)
            room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, kernel)
            room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, kernel)
        
        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            room_mask, connectivity=8
        )
        
        rooms = []
        min_area = self.config['postprocess'].get('min_area', 100)
        
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                room_mask_i = (labels == i).astype(np.uint8)
                rooms.append({
                    'id': i,
                    'mask': room_mask_i,
                    'area_pixels': area,
                    'centroid': centroids[i],
                    'bbox': stats[i, :4]  # x, y, w, h
                })
        
        return rooms
    
    def visualize(self, image_path, mask, colored_mask, save_path=None):
        """
        Visualize segmentation results
        
        Args:
            image_path: Original image path
            mask: Segmentation mask
            colored_mask: Colored mask
            save_path: Save path
        """
        # Read original image
        image = cv2.imread(str(image_path))
        
        # Overlay mask
        overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)
        
        if save_path:
            cv2.imwrite(save_path, overlay)
            print(f"Visualization result saved to: {save_path}")
        
        return overlay

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment rooms in floor plan')
    parser.add_argument('--image', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--model', type=str,
                       default='models/segmentation/best_model.pth',
                       help='Model path')
    parser.add_argument('--config', type=str,
                       default='config/room_segmentation.yaml',
                       help='Configuration file path')
    parser.add_argument('--save', type=str, default=None,
                       help='Save path for visualization result')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file does not exist: {args.model}")
        print("Please train the model first or check the model path!")
        return
    
    # Create segmenter
    segmenter = RoomSegmenter(args.model, args.config)
    
    # Execute segmentation
    print(f"Segmenting: {args.image}")
    mask, colored_mask = segmenter.segment(args.image)
    
    # Extract rooms
    rooms = segmenter.extract_rooms(mask)
    
    print(f"\nDetected {len(rooms)} rooms")
    for i, room in enumerate(rooms, 1):
        print(f"Room {i}: area = {room['area_pixels']} pixels")
    
    # Visualize
    if args.save:
        segmenter.visualize(args.image, mask, colored_mask, args.save)

if __name__ == '__main__':
    main()
