# Baseline Comparison Experiments

Reproducible scripts for the comparative evaluation section in `Paper.txt`.

## Segmentation: U-Net vs DeepLabv3+

Same train/val split (seed=42, 85/15), same loss, optimizer, and epochs as `train_segmentation_improved.py`:

```powershell
.\venv\Scripts\activate
python scripts/compare_segmentation_baselines.py ^
  --images data/segmentation_augmented/images ^
  --masks data/segmentation_augmented/masks ^
  --epochs 100 --architectures unet deeplabv3plus
```

Results: `models/baseline_comparison/segmentation_results.json`

## Detection: YOLOv8 vs Faster R-CNN

YOLOv8 已完成时，若 Faster R-CNN 卡在下载 COCO 权重，请用 `--frcnn-pretrained none` 跳过下载：

```powershell
.\venv\Scripts\activate
python scripts\compare_detection_baselines.py ^
  --data-yaml data/train_90/data.yaml ^
  --epochs 100 --models fasterrcnn ^
  --skip-yolov8 --frcnn-pretrained none
```

或直接双击 `run_frcnn_baseline.bat`。

若网络正常、需要 COCO 预训练权重，将 `--frcnn-pretrained none` 改为 `coco`，或手动下载后指定：

```powershell
# 可选：手动下载 (~160MB) 放到 models/baseline_comparison/weights/
python scripts\compare_detection_baselines.py ... --frcnn-weights models/baseline_comparison/weights/fasterrcnn_resnet50_fpn_coco.pth
```

Results: `models/baseline_comparison/detection_results.json`

## CubiCasa5K Cross-Dataset Evaluation

1. Download CubiCasa5K from [Zenodo](https://zenodo.org/record/2613548) to `data/CubiCasa5k/`
2. Map polygon labels to our 5-class schema (background, wall, room, door_area, window_area)
3. Fine-tune with ImageNet-pretrained encoders for 50 epochs:

```powershell
python scripts/compare_segmentation_baselines.py ^
  --images data/CubiCasa5k/processed/images ^
  --masks data/CubiCasa5k/processed/masks ^
  --encoder-weights imagenet --epochs 50
```

## Updating Paper Tables

After running experiments, update Table `tab:comparison_detection`, `tab:comparison_segmentation`, and `tab:cubicasa_comparison` in `Paper.txt` with values from the JSON result files.
