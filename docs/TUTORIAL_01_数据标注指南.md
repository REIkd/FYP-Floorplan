# 教程 1: 数据标注指南

数据标注是训练模型最关键的步骤。本教程将指导您如何标注平面图数据。

## 目录
1. [目标检测标注（家具图例）](#1-目标检测标注)
2. [语义分割标注（房间分割）](#2-语义分割标注)

---

## 1. 目标检测标注（家具图例）

### 1.1 安装标注工具

推荐使用 **LabelImg** 进行YOLO格式标注。

```bash
pip install labelImg
```

或者使用在线工具 [Roboflow](https://roboflow.com/)

### 1.2 运行LabelImg

```bash
labelImg data/images data/labels_detection
```

### 1.3 标注流程

1. **打开图片**: 点击 "Open Dir" 选择 `data/images`
2. **设置保存目录**: 点击 "Change Save Dir" 选择 `data/labels_detection`
3. **选择YOLO格式**: View → YOLO
4. **开始标注**:
   - 按 `W` 键创建矩形框
   - 框选家具图例
   - 选择类别（如 door, window, table等）
   - 按 `Ctrl+S` 保存
   - 按 `D` 键切换到下一张图片

### 1.4 需要标注的类别

建议标注以下常见类别（可根据您的数据调整）：

**基础类别**:
- door (门)
- window (窗)
- wall (墙角标记)

**家具类别**:
- bed (床)
- sofa (沙发)
- table (桌子)
- chair (椅子)
- desk (书桌)

**厨卫类别**:
- toilet (马桶)
- sink (水槽)
- bathtub (浴缸)
- stove (炉灶)
- refrigerator (冰箱)

**其他**:
- wardrobe (衣柜)
- bookshelf (书架)
- plant (植物)
- tv (电视)

### 1.5 标注技巧

1. **框要紧凑**: 尽量贴合物体边缘
2. **避免遗漏**: 确保每个家具图例都被标注
3. **类别一致**: 相同的物体使用相同的类别
4. **检查重复**: 避免同一物体被标注多次

### 1.6 标注示例

```
平面图示例:
┌──────────────────────┐
│  [门]  房间1  [窗]   │
│  [桌子]    [椅子]    │
│                      │
│  [门]  房间2  [床]   │
└──────────────────────┘

标注文件 (YOLO格式 - room.txt):
0 0.1 0.3 0.05 0.08   # door (class_id, x_center, y_center, width, height)
1 0.8 0.3 0.1 0.05    # window
2 0.3 0.5 0.15 0.15   # table
3 0.6 0.5 0.08 0.08   # chair
0 0.1 0.8 0.05 0.08   # door
4 0.7 0.8 0.2 0.15    # bed
```

### 1.7 质量检查

标注完成后，检查：
- ✓ 每张图片都有对应的`.txt`文件
- ✓ 标注框位置正确
- ✓ 类别标签正确
- ✓ 没有重复标注

可以使用我们提供的检查工具：

```bash
python src/utils/prepare_dataset.py --images-dir data/images --check-labels data/labels_detection
```

---

## 2. 语义分割标注（房间分割）

### 2.1 安装标注工具

推荐使用 **LabelMe** 进行多边形标注。

```bash
pip install labelme
```

或使用在线工具 [CVAT](https://cvat.org/)

### 2.2 运行LabelMe

```bash
labelme data/images --output data/labels_segmentation --labels labels.txt
```

创建 `labels.txt` 文件，内容如下：
```
background
wall
room
door_area
window_area
```

### 2.3 标注流程

1. **创建多边形**: 
   - 点击 "Create Polygons"
   - 沿着房间边界点击标点
   - 右键或双击完成多边形
   - 选择类别 "room"

2. **标注墙壁**:
   - 同样方式标注墙壁
   - 选择类别 "wall"

3. **标注门窗区域**:
   - 标注门的位置为 "door_area"
   - 标注窗的位置为 "window_area"

4. **保存**: 
   - Ctrl+S 保存为JSON
   - 后续需要转换为mask图片

### 2.4 转换JSON为Mask

LabelMe保存的是JSON格式，需要转换为PNG mask：

```python
import json
import numpy as np
from labelme import utils
import cv2

def json_to_mask(json_path, output_path, label_name_to_value):
    """将LabelMe JSON转换为mask"""
    with open(json_path) as f:
        data = json.load(f)
    
    img_shape = (data['imageHeight'], data['imageWidth'])
    mask = np.zeros(img_shape, dtype=np.uint8)
    
    for shape in data['shapes']:
        label = shape['label']
        points = np.array(shape['points'], dtype=np.int32)
        
        label_value = label_name_to_value.get(label, 0)
        cv2.fillPoly(mask, [points], label_value)
    
    cv2.imwrite(output_path, mask)

# 使用示例
label_map = {
    'background': 0,
    'wall': 1,
    'room': 2,
    'door_area': 3,
    'window_area': 4
}

json_to_mask('data/labels_segmentation/image1.json', 
             'data/labels_segmentation/masks/image1.png',
             label_map)
```

### 2.5 标注技巧

1. **优先标注房间**: 先标注所有房间区域
2. **再标注墙壁**: 墙壁通常是连接房间的边界
3. **门窗准确**: 门窗位置要准确，因为影响房间连通性
4. **避免重叠**: 不同类别区域不要重叠

### 2.6 简化标注方案

如果时间有限，可以采用简化方案：
- **只标注房间和墙壁** (2类)
- 使用图像处理方法自动提取墙壁

### 2.7 标注数量建议

- **最少**: 50-100张图片
- **推荐**: 150-200张图片
- **理想**: 300张以上

可以采用**数据增强**减少标注量。

---

## 3. 标注时间估算

- **目标检测**: 每张图片 5-10分钟
- **语义分割**: 每张图片 10-20分钟

标注100张图片大约需要：
- 目标检测: 10-15小时
- 语义分割: 15-30小时

## 4. 标注质量控制

### 4.1 多人标注

如果多人协作标注，需要：
1. 统一标注规范
2. 定期检查一致性
3. 交叉验证标注结果

### 4.2 标注评审

建议流程：
1. 初次标注
2. 自我检查
3. 交叉评审
4. 修正错误
5. 最终确认

## 5. 下一步

标注完成后，继续下一个教程：
- [教程2: 模型训练](TUTORIAL_02_模型训练.md)

---

## 常见问题

**Q: 标注太耗时怎么办？**
A: 可以先标注一小部分（50张），训练初步模型，然后使用模型预标注，人工校正。

**Q: 图片质量差怎么办？**
A: 可以先进行图像预处理（去噪、增强对比度）再标注。

**Q: 某些图例不清楚是什么怎么办？**
A: 跳过不确定的，或创建"unknown"类别，后续统一处理。

**Q: 要不要标注文字标签？**
A: 如果您需要提取文字信息（如房间名称），可以单独使用OCR，不在本项目范围内。

---

**下一步**: [模型训练教程 →](TUTORIAL_02_模型训练.md)

