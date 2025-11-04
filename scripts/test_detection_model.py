#!/usr/bin/env python3
"""
Object Detection Model Testing Tool
"""

import argparse
from ultralytics import YOLO
from pathlib import Path
import os
from collections import Counter

def test_single_image(model_path, image_path, conf_threshold=0.25, save_results=True):
    """Test single image"""
    
    print("=" * 80)
    print("Floor Plan Furniture Detection - Single Image Test")
    print("=" * 80)
    print()
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Number of classes: {len(model.names)}")
    print(f"Classes: {', '.join(model.names.values())}")
    print()
    
    # Predict
    print(f"Test image: {image_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print()
    
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        save=save_results,
        save_txt=True,
        save_conf=True,
        project='runs/detect',
        name='test',
        exist_ok=True
    )
    
    # Analyze results
    result = results[0]
    
    print("=" * 80)
    print("Detection Results")
    print("=" * 80)
    print()
    
    if len(result.boxes) == 0:
        print("[X] No objects detected")
    else:
        print(f"[OK] Detected {len(result.boxes)} objects\n")
        
        # Statistics by class
        class_counts = Counter()
        class_details = {}
        
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            
            class_counts[class_name] += 1
            if class_name not in class_details:
                class_details[class_name] = []
            class_details[class_name].append(conf)
        
        print("Detailed detection:")
        for class_name in sorted(class_details.keys()):
            confs = class_details[class_name]
            avg_conf = sum(confs) / len(confs)
            print(f"  {class_name}:")
            print(f"    Count: {len(confs)}")
            print(f"    Avg. confidence: {avg_conf:.2%}")
            print(f"    Confidence range: {min(confs):.2%} - {max(confs):.2%}")
    
    if save_results:
        print()
        print("Results saved to:")
        print(f"  Images: runs/detect/test/")
        print(f"  Labels: runs/detect/test/labels/")
    
    print()
    return results


def test_batch_images(model_path, images_dir, conf_threshold=0.25):
    """Batch test images"""
    
    print("=" * 80)
    print("Floor Plan Furniture Detection - Batch Test")
    print("=" * 80)
    print()
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Get all images
    image_files = list(Path(images_dir).glob('*.jpg'))
    image_files.extend(list(Path(images_dir).glob('*.png')))
    
    print(f"Found {len(image_files)} images")
    print()
    
    # Batch prediction
    results = model.predict(
        source=images_dir,
        conf=conf_threshold,
        save=True,
        save_txt=True,
        project='runs/detect',
        name='test_batch',
        exist_ok=True
    )
    
    # Statistics
    total_detections = 0
    class_counts_total = Counter()
    
    for result in results:
        total_detections += len(result.boxes)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            class_counts_total[class_name] += 1
    
    print("=" * 80)
    print("Batch Test Results")
    print("=" * 80)
    print()
    print(f"Total detections: {total_detections}")
    print(f"Average per image: {total_detections/len(image_files):.1f} objects")
    print()
    print("Class statistics:")
    for class_name, count in class_counts_total.most_common():
        print(f"  {class_name}: {count}")
    
    print()
    print(f"Results saved to: runs/detect/test_batch/")
    print()


def evaluate_on_val(model_path, data_yaml):
    """Evaluate on validation set"""
    
    print("=" * 80)
    print("Validation Set Evaluation")
    print("=" * 80)
    print()
    
    model = YOLO(model_path)
    
    print("Evaluating...")
    metrics = model.val(
        data=data_yaml,
        split='val',
        save_json=True,
        save_hybrid=True
    )
    
    print()
    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print()
    print(f"mAP50: {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)")
    print(f"mAP50-95: {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)")
    print(f"Precision: {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)")
    print(f"Recall: {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)")
    print()
    
    # Per-class results
    if hasattr(metrics.box, 'ap_class_index'):
        print("Per-class AP50:")
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            class_name = model.names[int(cls_idx)]
            ap = metrics.box.ap50[i]
            print(f"  {class_name}: {ap:.4f} ({ap*100:.2f}%)")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Test object detection model')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt',
                        help='Model path')
    parser.add_argument('--image', type=str,
                        help='Single image path')
    parser.add_argument('--dir', type=str,
                        help='Image directory (batch test)')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate on validation set')
    parser.add_argument('--data', type=str, default='data/train_53/data.yaml',
                        help='Data configuration file')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        return
    
    if args.eval:
        # Validation set evaluation
        evaluate_on_val(args.model, args.data)
    elif args.image:
        # Single image test
        if not os.path.exists(args.image):
            print(f"Error: Image does not exist: {args.image}")
            return
        test_single_image(args.model, args.image, args.conf)
    elif args.dir:
        # Batch test
        if not os.path.exists(args.dir):
            print(f"Error: Directory does not exist: {args.dir}")
            return
        test_batch_images(args.model, args.dir, args.conf)
    else:
        # Default: use first image from validation set
        val_dir = 'data/train_53/images/val'
        if os.path.exists(val_dir):
            image_files = list(Path(val_dir).glob('*.jpg'))
            if image_files:
                print("No test image specified, using first image from validation set\n")
                test_single_image(args.model, str(image_files[0]), args.conf)
            else:
                print("Validation set images not found")
        else:
            print("Please specify test method using --image, --dir, or --eval")
            print("\nExamples:")
            print("  python scripts/test_detection_model.py --image test.jpg")
            print("  python scripts/test_detection_model.py --dir data/test_images")
            print("  python scripts/test_detection_model.py --eval")


if __name__ == '__main__':
    main()
