# Model Weights (GitHub)

Files tracked in this repo for **out-of-the-box web app deployment**:

| File | Size | Used by |
|------|------|---------|
| `detection/yolov8_best.pt` | ~6 MB | `app.py` — YOLOv8 detection |
| `segmentation/best_model.pth` | ~93 MB | `app.py` — U-Net segmentation |

## Not uploaded (GitHub limits / reproducible)

| File | Size | Reason |
|------|------|--------|
| `baseline_comparison/fasterrcnn_best.pth` | **158 MB** | Exceeds GitHub **100 MB** file limit |
| `baseline_comparison/deeplabv3plus_best.pth` | ~86 MB | Baseline only; regenerate with `scripts/compare_segmentation_baselines.py` |
| `segmentation/quick_test_model.pth` | ~55 MB | Dev checkpoint; not deployed |
| `runs/detect/*/weights/*.pt` | ~6 MB each | Ignored; copy to `detection/yolov8_best.pt` after training |

## Regenerate baselines locally

```powershell
.\venv\Scripts\activate
python scripts\compare_segmentation_baselines.py --images data/segmentation_augmented/images --masks data/segmentation_augmented/masks --epochs 100
python scripts\compare_detection_baselines.py --data-yaml data/train_90/data.yaml --epochs 100
```

Results JSON: `models/baseline_comparison/*_results.json` (small; safe to commit if generated).

## If push still fails

GitHub warns above **50 MB** and **rejects** above **100 MB**. Use [Git LFS](https://git-lfs.github.com/) for single files over 100 MB, or host weights on Zenodo/Google Drive and link in the main README.
