@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ================================================================================
echo Faster R-CNN Baseline (skip download, reuse YOLOv8 results)
echo ================================================================================
echo.
echo YOLOv8 already trained. This runs Faster R-CNN only (--frcnn-pretrained none).
echo CPU training is slow; expect 1-3 hours for 30 epochs.
echo.

call venv\Scripts\activate.bat

python scripts\compare_detection_baselines.py ^
  --data-yaml data/train_90/data.yaml ^
  --epochs 100 ^
  --models fasterrcnn ^
  --skip-yolov8 ^
  --frcnn-pretrained none

echo.
echo Done. Results: models\baseline_comparison\detection_results.json
pause
