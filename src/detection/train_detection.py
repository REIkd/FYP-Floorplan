"""
Furniture Detection Model Training Script
Using YOLOv8 for furniture symbol detection in floor plans
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml
import torch

def train_furniture_detector(config_path, model_size='n', resume=False):
    """
    Train furniture detection model
    
    Args:
        config_path: Configuration file path
        model_size: Model size (n, s, m, l, x)
        resume: Whether to resume from last training
    """
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    if resume and os.path.exists('runs/detect/train/weights/last.pt'):
        print("Resuming from last training...")
        model = YOLO('runs/detect/train/weights/last.pt')
    else:
        print(f"Initializing YOLOv8{model_size} model...")
        model = YOLO(f'yolov8{model_size}.pt')
    
    # Train model
    results = model.train(
        data=config_path,
        epochs=config.get('epochs', 100),
        batch=config.get('batch', 16),
        imgsz=config.get('imgsz', 640),
        device=device,
        project='runs/detect',
        name='furniture_detection',
        exist_ok=True,
        pretrained=True,
        optimizer=config.get('optimizer', 'AdamW'),
        lr0=config.get('lr0', 0.001),
        weight_decay=config.get('weight_decay', 0.0005),
        save=True,
        save_period=10,  # Save every 10 epochs
        plots=True,
        verbose=True
    )
    
    # Validate model
    print("\nStarting model validation...")
    metrics = model.val()
    
    print(f"\nTraining complete!")
    print(f"Best model saved at: runs/detect/furniture_detection/weights/best.pt")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return model, results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train furniture detection model')
    parser.add_argument('--config', type=str, 
                       default='config/furniture_detection.yaml',
                       help='Configuration file path')
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last training')
    
    args = parser.parse_args()
    
    # Train model
    model, results = train_furniture_detector(
        args.config,
        args.model_size,
        args.resume
    )

if __name__ == '__main__':
    main()
