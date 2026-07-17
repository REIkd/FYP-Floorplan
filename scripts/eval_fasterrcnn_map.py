#!/usr/bin/env python3
"""Evaluate Faster R-CNN with COCO-style mAP50 and mAP50-95."""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_detection_baselines import YoloDetectionDataset, collate_fn


def evaluate_map(model, dataset, device, score_thresh=0.05):
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        class_metrics=False,
        backend='faster_coco_eval',
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            pred = outputs[0]
            gt = targets[0]

            keep = pred['scores'] >= score_thresh
            preds = {
                'boxes': pred['boxes'][keep].cpu(),
                'scores': pred['scores'][keep].cpu(),
                'labels': pred['labels'][keep].cpu(),
            }
            gts = {
                'boxes': gt['boxes'].cpu(),
                'labels': gt['labels'].cpu(),
            }
            metric.update([preds], [gts])

    result = metric.compute()
    map50 = float(result['map_50'].item())
    map5095 = float(result['map'].item())
    mar = float(result['mar_100'].item()) if 'mar_100' in result else 0.0

    return {
        'mAP50': map50,
        'mAP50_95': map5095,
        'recall': mar,
        'precision': None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-yaml', default='data/train_90/data.yaml')
    parser.add_argument('--weights', default='models/baseline_comparison/fasterrcnn_best.pth')
    parser.add_argument('--output', default='models/baseline_comparison/detection_results.json')
    args = parser.parse_args()

    with open(args.data_yaml, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg['path'])
    val_img = base / cfg['val']
    val_lbl = base / 'labels' / 'val'
    num_classes = cfg['nc']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.to(device)

    val_ds = YoloDetectionDataset(val_img, val_lbl, [str(i) for i in range(num_classes)])
    metrics = evaluate_map(model, val_ds, device)
    print(json.dumps(metrics, indent=2))

    output = Path(args.output)
    results = json.loads(output.read_text(encoding='utf-8')) if output.exists() else []

    csv_path = Path('models/baseline_comparison/yolov8/baseline/results.csv')
    if csv_path.exists():
        last = csv_path.read_text(encoding='utf-8').strip().splitlines()[-1].split(',')
        yolo = {
            'architecture': 'yolov8n',
            'mAP50': float(last[7]),
            'mAP50_95': float(last[8]),
            'precision': float(last[5]),
            'recall': float(last[6]),
            'training_time_s': float(last[1]),
        }
        results = [r for r in results if r.get('architecture') != 'yolov8n']
        results.insert(0, yolo)

    frcnn = next((r for r in results if r.get('architecture') == 'fasterrcnn'), {})
    frcnn.update({
        'architecture': 'fasterrcnn',
        'mAP50': metrics['mAP50'],
        'mAP50_95': metrics['mAP50_95'],
        'recall': metrics['recall'],
    })
    if metrics['precision'] is not None:
        frcnn['precision'] = metrics['precision']
    results = [r for r in results if r.get('architecture') != 'fasterrcnn'] + [frcnn]
    results.sort(key=lambda x: 0 if x['architecture'] == 'yolov8n' else 1)

    output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Updated {output}')


if __name__ == '__main__':
    main()
