@echo off
chcp 65001 >nul
cd /d "%~dp0"

set IMG_DIR=E:\Project\FYP-Floorplan\data\train_90\images\train
set IMG_FILE=%IMG_DIR%\FloorPlan-3-_jpg.rf.2f119720ab892966bae33f3fdece8945.jpg
set CLASS_FILE=E:\Project\FYP-Floorplan\data\train_90\labels\train\predefined_classes.txt
set SAVE_DIR=E:\Project\FYP-Floorplan\data\train_90\labels\train

echo ================================================================================
echo LabelImg - FloorPlan-3
echo ================================================================================
echo.
echo 参数顺序: 图片/目录  类别文件  保存目录
echo.
echo 图片: %IMG_FILE%
echo 类别: %CLASS_FILE%
echo 保存: %SAVE_DIR%
echo.

REM 用法: labelImg image classFile saveDir  (NOT image saveDir classFile)
labelImg "%IMG_FILE%" "%CLASS_FILE%" "%SAVE_DIR%"

pause
