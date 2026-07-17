#!/usr/bin/env python3
"""
Compare detection baselines on the same YOLO-format dataset.

Models:
  - yolov8n      : YOLOv8-nano (proposed, via Ultralytics)
  - fasterrcnn   : Faster R-CNN ResNet50-FPN (via torchvision)

Results saved to models/baseline_comparison/detection_results.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor
from ultralytics import YOLO


class YoloDetectionDataset(Dataset):
    """Minimal YOLO-format dataset for Faster R-CNN training."""

    def __init__(self, images_dir, labels_dir, class_names, img_size=640):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names
        self.img_size = img_size
        self.samples = []
        for img_path in sorted(self.images_dir.glob('*')):
            if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            label_path = self.labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2

        img_path, label_path = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        boxes, labels = [], []
        for line in label_path.read_text(encoding='utf-8').strip().splitlines():
            if not line.strip():
                continue
            cls, xc, yc, bw, bh = map(float, line.split())
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls) + 1)  # 0 reserved for background

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
        }
        return to_tensor(image), target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def load_fasterrcnn(num_classes, weights_path=None, use_pretrained=True):
    """Load Faster R-CNN, preferring a local weights file to avoid slow hub downloads."""
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    local_candidates = [
        Path(weights_path) if weights_path else None,
        Path('models/baseline_comparison/weights/fasterrcnn_resnet50_fpn_coco.pth'),
        Path.home() / '.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    ]

    model = None
    for candidate in local_candidates:
        if candidate and candidate.exists() and candidate.stat().st_size > 100_000_000:
            print(f"Loading Faster R-CNN weights from local file: {candidate}")
            model = fasterrcnn_resnet50_fpn(weights=None)
            state = torch.load(candidate, map_location='cpu')
            model.load_state_dict(state, strict=False)
            break

    if model is None and use_pretrained:
        print("Downloading Faster R-CNN COCO weights (may take several minutes)...")
        print("Tip: if stuck, cancel and run with --frcnn-pretrained none")
        try:
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        except Exception as exc:
            print(f"Pretrained download failed ({exc}); training from scratch.")
            model = fasterrcnn_resnet50_fpn(weights=None)
    elif model is None:
        print("Training Faster R-CNN from scratch (no pretrained weights).")
        model = fasterrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def train_fasterrcnn(train_dir, val_dir, labels_train, labels_val, num_classes,
                     epochs, lr, batch_size, output_dir,
                     weights_path=None, use_pretrained=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = [str(i) for i in range(num_classes)]

    train_ds = YoloDetectionDataset(train_dir, labels_train, class_names)
    val_ds = YoloDetectionDataset(val_dir, labels_val, class_names)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    model = load_fasterrcnn(num_classes, weights_path, use_pretrained).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training Faster R-CNN on {len(train_ds)} images...")
    start = time.time()
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Faster R-CNN epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / 'fasterrcnn_best.pth')

    elapsed = time.time() - start
    return {
        'architecture': 'fasterrcnn',
        'train_loss': best_loss,
        'training_time_s': elapsed,
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'note': 'Use YOLO val metrics for mAP comparison after export; '
                'Faster R-CNN mAP evaluated via separate COCO eval if needed.',
    }


def train_yolov8(data_yaml, epochs, batch_size, img_size, output_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n.pt')
    start = time.time()
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=str(Path(output_dir) / 'yolov8'),
        name='baseline',
        exist_ok=True,
        verbose=True,
    )
    metrics = model.val(data=data_yaml, split='val')
    elapsed = time.time() - start
    return {
        'architecture': 'yolov8n',
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'training_time_s': elapsed,
    }


def yolov8_results_from_csv(output_dir):
    """Reuse completed YOLOv8 run if training already finished."""
    csv_path = Path(output_dir) / 'yolov8' / 'baseline' / 'results.csv'
    if not csv_path.exists():
        return None
    last = csv_path.read_text(encoding='utf-8').strip().splitlines()[-1].split(',')
    return {
        'architecture': 'yolov8n',
        'mAP50': float(last[7]),
        'mAP50_95': float(last[8]),
        'precision': float(last[5]),
        'recall': float(last[6]),
        'training_time_s': float(last[1]),
        'source': str(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(description='Detection baseline comparison')
    parser.add_argument('--data-yaml', default='data/train_90/data.yaml')
    parser.add_argument('--models', nargs='+', default=['yolov8n', 'fasterrcnn'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--output-dir', default='models/baseline_comparison')
    parser.add_argument('--frcnn-pretrained', default='none',
                        choices=['coco', 'none'],
                        help='Faster R-CNN init: coco (download) or none (from scratch, no download)')
    parser.add_argument('--frcnn-weights', default=None,
                        help='Local .pth path for Faster R-CNN COCO weights')
    parser.add_argument('--skip-yolov8', action='store_true',
                        help='Skip YOLOv8; reuse results.csv if available')
    args = parser.parse_args()

    with open(args.data_yaml, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg['path'])
    train_img = base / cfg['train']
    val_img = base / cfg['val']
    train_lbl = base / 'labels' / 'train'
    val_lbl = base / 'labels' / 'val'
    num_classes = cfg['nc']

    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'yolov8n' in args.models and not args.skip_yolov8:
        print('=' * 80)
        print('Training YOLOv8-nano baseline')
        print('=' * 80)
        results.append(train_yolov8(
            args.data_yaml, args.epochs, args.batch_size, args.img_size, args.output_dir
        ))
    elif 'yolov8n' in args.models:
        cached = yolov8_results_from_csv(args.output_dir)
        if cached:
            print('Reusing completed YOLOv8 results from results.csv')
            results.append(cached)
        else:
            print('Warning: --skip-yolov8 set but results.csv not found; skipping YOLOv8.')

    if 'fasterrcnn' in args.models:
        print('=' * 80)
        print('Training Faster R-CNN baseline')
        print('=' * 80)
        frcnn_epochs = min(args.epochs, 30)  # Faster R-CNN is slower; cap for practicality
        results.append(train_fasterrcnn(
            train_img, val_img, train_lbl, val_lbl, num_classes,
            frcnn_epochs, args.lr, max(2, args.batch_size // 2), args.output_dir,
            weights_path=args.frcnn_weights,
            use_pretrained=(args.frcnn_pretrained == 'coco'),
        ))
        # Evaluate YOLO-format mAP via conversion is non-trivial; store training metrics
        results[-1]['epochs'] = frcnn_epochs

    output_path = output_dir / 'detection_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 80}\nDetection Baseline Summary\n{'=' * 80}")
    for item in results:
        if 'mAP50' in item:
            print(f"{item['architecture']:12s}  mAP50={item['mAP50']*100:.2f}%  "
                  f"mAP50-95={item['mAP50_95']*100:.2f}%")
        else:
            print(f"{item['architecture']:12s}  final_loss={item['train_loss']:.4f}")
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
