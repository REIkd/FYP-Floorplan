# 教程 3: 使用平面图分析Agent

本教程将指导您如何使用训练好的模型进行平面图分析。

## 前提条件

- ✓ 已完成模型训练（参考[教程2](TUTORIAL_02_模型训练.md)）
- ✓ 有训练好的检测和分割模型

---

## 第一部分: 基础使用

### 1.1 快速开始

最简单的使用方式：

```bash
python src/agent/floorplan_agent.py \
    --image data/images/example.jpg \
    --scale "1:100" \
    --reference-pixels 200 \
    --reference-length 200
```

**参数说明**:
- `--image`: 要分析的平面图图片
- `--scale`: 比例尺（如 1:100）
- `--reference-pixels`: 参考线段的像素长度
- `--reference-length`: 参考线段的实际长度(cm)

### 1.2 理解比例尺和校准

平面图分析中最关键的是**比例尺校准**。

#### 方法1: 使用图上标注的比例尺

如果平面图上有比例尺标注（如"1:100"）：

1. 在图上找一条已知长度的线段（比如标注为2米的墙）
2. 用图片查看工具测量这条线段的像素长度（假设是200像素）
3. 转换：2米 = 200厘米

```bash
--scale "1:100" \
--reference-pixels 200 \
--reference-length 200
```

#### 方法2: 使用已知尺寸

如果知道某个房间的实际尺寸：

1. 在图上测量这个房间的像素宽度（假设400像素）
2. 假设实际宽度是4米（400厘米）

```bash
--reference-pixels 400 \
--reference-length 400
```

#### 如何测量像素长度？

**Windows**:
```bash
# 使用Paint
1. 打开图片
2. 用"线条"工具画一条线
3. 查看右下角的像素坐标差值
```

**Python脚本**:
```python
import cv2

def measure_distance(image_path):
    """交互式测量像素距离"""
    img = cv2.imread(image_path)
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            if len(points) == 2:
                dist = ((points[1][0]-points[0][0])**2 + 
                       (points[1][1]-points[0][1])**2)**0.5
                print(f"像素距离: {dist:.2f}")
                points.clear()
    
    cv2.namedWindow('Measure')
    cv2.setMouseCallback('Measure', mouse_callback)
    
    while True:
        cv2.imshow('Measure', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

# 使用
measure_distance('data/images/example.jpg')
```

---

## 第二部分: 详细使用

### 2.1 完整命令示例

```bash
python src/agent/floorplan_agent.py \
    --image data/images/floor_plan_01.jpg \
    --detection-model runs/detect/furniture_detection/weights/best.pt \
    --segmentation-model models/segmentation/best_model.pth \
    --scale "1:100" \
    --reference-pixels 200 \
    --reference-length 200 \
    --unit m2 \
    --save-dir output/analysis_01 \
    --json output/analysis_01/results.json
```

### 2.2 输出结果

运行后会生成：

**1. 终端输出**:
```
============================================================
开始分析平面图: data/images/floor_plan_01.jpg
============================================================

【步骤 1/3】 家具图例检测
------------------------------------------------------------
==================================================
家具检测统计
==================================================
检测到的家具总数: 23

各类家具数量:
--------------------------------------------------
  door                :   3个
  window              :   5个
  bed                 :   2个
  table               :   1个
  chair               :   4个
  sofa                :   1个
  toilet              :   1个
  sink                :   2个
==================================================

✓ 家具检测完成

【步骤 2/3】 房间语义分割
------------------------------------------------------------
✓ 检测到 4 个房间
✓ 房间分割完成

【步骤 3/3】 面积计算
------------------------------------------------------------
比例尺设置为: 1:100
校准完成: 200.0 像素 = 200.0 cm
每厘米像素数: 1.00

==================================================
房间面积统计
==================================================
房间  1:    25.60 m² (25600 像素)
房间  2:    18.30 m² (18300 像素)
房间  3:    12.50 m² (12500 像素)
房间  4:     8.70 m² (8700 像素)
--------------------------------------------------
总面积:      65.10 m²
==================================================

✓ 面积计算完成

============================================================
分析完成!
============================================================
```

**2. 可视化图片**:
- `output/analysis_01/floor_plan_01_furniture.jpg` - 家具检测结果
- `output/analysis_01/floor_plan_01_segmentation.jpg` - 房间分割结果

**3. JSON结果**:
```json
{
  "image_path": "data/images/floor_plan_01.jpg",
  "furniture": {
    "total": 23,
    "by_class": {
      "door": 3,
      "window": 5,
      "bed": 2,
      "table": 1,
      "chair": 4,
      "sofa": 1,
      "toilet": 1,
      "sink": 2
    }
  },
  "rooms": [
    {"id": 1, "area_pixels": 25600},
    {"id": 2, "area_pixels": 18300},
    {"id": 3, "area_pixels": 12500},
    {"id": 4, "area_pixels": 8700}
  ],
  "area": {
    "rooms": [
      {"id": 1, "area": 25.6, "area_pixels": 25600},
      {"id": 2, "area": 18.3, "area_pixels": 18300},
      {"id": 3, "area": 12.5, "area_pixels": 12500},
      {"id": 4, "area": 8.7, "area_pixels": 8700}
    ],
    "total_area": 65.1,
    "unit": "m2"
  },
  "success": true
}
```

---

## 第三部分: 批量处理

### 3.1 批量分析脚本

创建批量处理脚本 `scripts/batch_analyze.py`:

```python
"""批量分析多个平面图"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from agent.floorplan_agent import FloorPlanAgent
import json

def batch_analyze(image_dir, output_dir, scale, ref_pixels, ref_length):
    """批量分析"""
    agent = FloorPlanAgent()
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_all = []
    
    for img_path in image_dir.glob('*.jpg'):
        print(f"\n处理: {img_path.name}")
        
        try:
            result = agent.analyze(
                image_path=str(img_path),
                scale=scale,
                reference_pixels=ref_pixels,
                reference_length=ref_length,
                save_dir=str(output_dir / img_path.stem)
            )
            results_all.append(result)
        except Exception as e:
            print(f"错误: {e}")
            continue
    
    # 保存汇总结果
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)
    
    print(f"\n批量处理完成! 共处理 {len(results_all)} 张图片")
    print(f"汇总结果: {summary_path}")

if __name__ == '__main__':
    batch_analyze(
        image_dir='data/images',
        output_dir='output/batch_results',
        scale='1:100',
        ref_pixels=200,
        ref_length=200
    )
```

运行：
```bash
python scripts/batch_analyze.py
```

### 3.2 生成统计报告

```python
"""生成批量分析的统计报告"""
import json
import pandas as pd
import matplotlib.pyplot as plt

def generate_report(summary_json):
    """生成报告"""
    with open(summary_json) as f:
        results = json.load(f)
    
    # 提取数据
    data = []
    for r in results:
        if r.get('success'):
            data.append({
                'image': Path(r['image_path']).name,
                'furniture_count': r['furniture']['total'],
                'room_count': len(r['rooms']),
                'total_area': r['area']['total_area']
            })
    
    df = pd.DataFrame(data)
    
    # 统计
    print("=" * 50)
    print("批量分析统计报告")
    print("=" * 50)
    print(f"总图片数: {len(df)}")
    print(f"\n家具统计:")
    print(f"  平均每张图: {df['furniture_count'].mean():.1f} 个")
    print(f"  最多: {df['furniture_count'].max()} 个")
    print(f"  最少: {df['furniture_count'].min()} 个")
    print(f"\n房间统计:")
    print(f"  平均每张图: {df['room_count'].mean():.1f} 个")
    print(f"\n面积统计:")
    print(f"  平均面积: {df['total_area'].mean():.1f} m²")
    print(f"  最大面积: {df['total_area'].max():.1f} m²")
    print(f"  最小面积: {df['total_area'].min():.1f} m²")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    df['furniture_count'].hist(ax=axes[0], bins=20)
    axes[0].set_title('家具数量分布')
    
    df['room_count'].hist(ax=axes[1], bins=10)
    axes[1].set_title('房间数量分布')
    
    df['total_area'].hist(ax=axes[2], bins=20)
    axes[2].set_title('面积分布')
    
    plt.tight_layout()
    plt.savefig('output/statistics.png')
    print(f"\n图表已保存: output/statistics.png")

generate_report('output/batch_results/summary.json')
```

---

## 第四部分: 高级应用

### 4.1 集成到Web应用

使用FastAPI创建Web服务：

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil

app = FastAPI()
agent = FloorPlanAgent()

@app.post("/analyze")
async def analyze_floorplan(
    file: UploadFile = File(...),
    scale: str = "1:100",
    ref_pixels: float = 200,
    ref_length: float = 200
):
    """分析上传的平面图"""
    # 保存上传的文件
    temp_path = f"temp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 分析
    result = agent.analyze(
        temp_path,
        scale=scale,
        reference_pixels=ref_pixels,
        reference_length=ref_length
    )
    
    return JSONResponse(content=result)

# 运行: uvicorn webapp:app --reload
```

### 4.2 导出为ONNX

加速推理：

```python
# 导出YOLO模型
from ultralytics import YOLO
model = YOLO('runs/detect/furniture_detection/weights/best.pt')
model.export(format='onnx')

# 导出分割模型
import torch
seg_model = torch.load('models/segmentation/best_model.pth')
dummy_input = torch.randn(1, 3, 512, 512)
torch.onnx.export(seg_model, dummy_input, 'models/segmentation/model.onnx')
```

---

## 第五部分: 常见问题

### Q1: 检测结果不准确

**检查**:
1. 模型是否训练充分
2. 图片质量是否良好
3. 置信度阈值是否合适（默认0.25）

**调整置信度**:
```python
detector.detect(image_path, conf_threshold=0.3)  # 提高阈值
```

### Q2: 分割结果有误

**可能原因**:
1. 模型训练不足
2. 图片与训练集差异大
3. 后处理参数不合适

**调整后处理**:
```yaml
# config/room_segmentation.yaml
postprocess:
  min_area: 200  # 增加最小面积阈值
  morphology: true
```

### Q3: 面积计算不准

**核心**: 校准必须准确

**建议**:
1. 使用多个参考线段取平均
2. 选择较长的参考线段（减少误差）
3. 使用专业工具测量像素

### Q4: 处理速度慢

**优化**:
1. 使用更小的模型
2. 降低输入图片分辨率
3. 导出为ONNX格式
4. 使用GPU推理
5. 批量处理而非逐张处理

---

## 第六部分: 实际案例

### 案例1: 房地产应用

自动化分析户型图：
```bash
# 批量分析小区所有户型
python scripts/batch_analyze.py \
    --input estate_plans/ \
    --output estate_analysis/ \
    --scale "1:100"

# 生成对比报告
python scripts/compare_plans.py \
    --results estate_analysis/summary.json \
    --output estate_report.pdf
```

### 案例2: 室内设计

分析现有户型，提供设计建议：
```python
result = agent.analyze('client_plan.jpg', ...)

# 根据结果提供建议
if result['area']['total_area'] < 60:
    print("建议: 小户型，采用开放式设计")
if result['furniture']['by_class'].get('window', 0) < 2:
    print("建议: 采光不足，考虑增加照明")
```

---

## 总结

您现在应该能够：
- ✓ 使用Agent分析单张平面图
- ✓ 理解比例尺校准的重要性
- ✓ 批量处理多张图片
- ✓ 生成分析报告
- ✓ 解决常见问题

---

**上一步**: [← 模型训练教程](TUTORIAL_02_模型训练.md)

如有问题，请参考[常见问题FAQ](FAQ.md)或提Issue！

