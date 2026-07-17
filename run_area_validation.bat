@echo off
chcp 65001 >nul
cd /d "%~dp0"
call venv\Scripts\activate.bat

echo [1/3] Fill GT from masks (if needed)...
python scripts\batch_area_validation.py --fill-gt-from-masks

echo [2/3] Run batch area validation...
python scripts\batch_area_validation.py

echo [3/3] Update Paper.txt table...
python scripts\batch_area_validation.py --update-paper

echo Done. See models\area_validation\results.json
pause
