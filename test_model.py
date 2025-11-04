#!/usr/bin/env python3
"""
Test Trained Object Detection Model
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import os

# Load trained best model
model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(model_path)

print("=" * 80)
print("Floor Plan Furniture Detection Model - Test Tool")
print("=" * 80)
print(f"\nModel path: {model_path}")
print(f"Number of classes: {len(model.names)}")
print(f"Classes: {list(model.names.values())}")
print()

# Select test image
test_image = input("Enter test image path (leave empty to use validation set image): ").strip()

if not test_image:
    # Use an image from validation set
    val_images = list(Path('data/train_53/images/val').glob('*.jpg'))
    if val_images:
        test_image = str(val_images[0])
        print(f"Using validation set image: {test_image}")
    else:
        print("Validation set images not found")
        exit(1)

# Perform prediction
print("\nPerforming detection...")
results = model.predict(
    source=test_image,
    conf=0.25,  # Confidence threshold
    save=True,  # Save results
    save_txt=True,  # Save label files
    save_conf=True,  # Save confidence
    project='runs/detect',
    name='test',
    exist_ok=True
)

# Display results
result = results[0]
print("\n" + "=" * 80)
print("Detection Results")
print("=" * 80)

if len(result.boxes) == 0:
    print("No objects detected")
else:
    print(f"Detected {len(result.boxes)} objects:\n")
    
    # Count by class
    from collections import Counter
    class_counts = Counter()
    
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        class_counts[class_name] += 1
        
        print(f"  - {class_name}: confidence {conf:.2%}")
    
    print("\nClass statistics:")
    for class_name, count in class_counts.most_common():
        print(f"  {class_name}: {count} objects")

print("\nResults saved to:")
print(f"  Images: runs/detect/test/*.jpg")
print(f"  Labels: runs/detect/test/labels/*.txt")
print()
