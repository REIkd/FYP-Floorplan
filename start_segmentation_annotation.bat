@echo off
chcp 65001 >nul
echo ================================================================================
echo LabelMe 房间分割标注工具
echo ================================================================================
echo.
echo 标注目标：
echo   - wall          墙壁
echo   - room          房间区域  
echo   - door_area     门的位置
echo   - window_area   窗的位置
echo.
echo 标注步骤：
echo   1. 使用 Create Polygons (Ctrl+N) 创建多边形
echo   2. 沿着墙壁/房间边界点击
echo   3. 双击完成多边形
echo   4. 输入标签名称
echo   5. 按 Ctrl+S 保存
echo   6. 按 D 键切换到下一张图片
echo.
echo 详细指南请查看：docs\房间分割标注指南.md
echo.
echo 图片目录：data\images_original
echo 保存目录：data\labels_segmentation\json
echo.
echo ================================================================================
echo 正在启动 LabelMe...
echo ================================================================================
echo.

labelme data\images_original --output data\labels_segmentation\json --labels config\segmentation_classes.txt --nodata

echo.
echo ================================================================================
echo LabelMe 已关闭
echo ================================================================================
echo.
echo 标注文件已保存到：data\labels_segmentation\json
echo.
echo 下一步：
echo   1. 转换标注为 mask：python scripts\convert_labelme_to_masks.py
echo   2. 训练分割模型：python src\segmentation\train_segmentation.py
echo.
pause

