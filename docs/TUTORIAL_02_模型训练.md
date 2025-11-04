# 教程 2: 模型训练指南

本教程将指导您如何训练家具检测和房间分割模型。

## 前提条件

- ✓ 已完成数据标注（参考[教程1](TUTORIAL_01_数据标注指南.md)）
- ✓ 已安装所有依赖 (`pip install -r requirements.txt`)
- ✓ 有GPU环境（推荐，CPU也可以但很慢）

---

## 第一部分: 家具检测模型训练

### 1.1 准备数据集

首先划分训练集、验证集和测试集：

```bash
python src/utils/prepare_dataset.py \
    --images-dir data/images \
    --output-dir data/splits \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1 \
    --check-labels data/labels_detection
```

输出：
```
找到 303 张图片
训练集: 212 张
验证集: 61 张
测试集: 30 张
✓ 所有文件都有对应的标注
```

### 1.2 检查配置文件

编辑 `config/furniture_detection.yaml`，确认类别设置：

```yaml
# 根据您的实际标注类别修改
names:
  0: door
  1: window
  2: table
  # ... 添加您标注的所有类别

nc: 16  # 类别数量，需要与实际匹配
```

### 1.3 开始训练

**选择模型大小**:
- `n` (nano) - 最快，精度较低，适合测试
- `s` (small) - 快速，精度中等，**推荐起步**
- `m` (medium) - 平衡，精度较高
- `l` (large) - 慢，精度高
- `x` (xlarge) - 最慢，精度最高

**训练命令**:

```bash
# 使用small模型（推荐）
python src/detection/train_detection.py \
    --config config/furniture_detection.yaml \
    --model-size s

# 如果中途中断，可以继续训练
python src/detection/train_detection.py \
    --config config/furniture_detection.yaml \
    --model-size s \
    --resume
```

### 1.4 训练过程监控

训练过程中会实时显示：
```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
1/100    3.14G     1.234      0.876      1.123      245         640
2/100    3.14G     1.156      0.812      1.089      238         640
...
```

也可以使用TensorBoard监控：

```bash
tensorboard --logdir runs/detect/furniture_detection
```

### 1.5 评估指标

训练完成后会显示：

```
mAP50: 0.85      # AP @ IoU=0.50
mAP50-95: 0.68   # AP @ IoU=0.50-0.95
```

**指标说明**:
- **mAP50 > 0.7**: 良好
- **mAP50 > 0.8**: 优秀
- **mAP50 > 0.9**: 极优

### 1.6 测试模型

```bash
python src/detection/detect_furniture.py \
    --image data/images/test_image.jpg \
    --model runs/detect/furniture_detection/weights/best.pt \
    --save output/test_detection.jpg \
    --json output/test_result.json
```

### 1.7 训练技巧

**如果精度不够**:
1. 增加训练epoch数（在配置文件中修改）
2. 使用更大的模型 (`m` 或 `l`)
3. 检查标注质量
4. 增加数据量
5. 调整学习率

**如果过拟合**:
1. 增加数据增强
2. 使用更小的模型
3. 添加正则化
4. 增加数据量

---

## 第二部分: 房间分割模型训练

### 2.1 准备分割数据

确保您的目录结构如下：

```
data/
├── images/              # 原始图片
├── labels_segmentation/
│   └── masks/          # PNG格式的mask图片
└── splits/             # 数据集划分
```

### 2.2 转换标注格式

如果您使用LabelMe标注，需要先转换JSON为mask：

```python
# 创建转换脚本
python scripts/convert_labelme_to_mask.py \
    --input data/labels_segmentation \
    --output data/labels_segmentation/masks
```

### 2.3 检查配置

编辑 `config/room_segmentation.yaml`:

```yaml
# 确认类别定义
classes:
  0: background
  1: wall
  2: room
  3: door_area
  4: window_area

num_classes: 5

# 根据您的GPU调整batch_size
training:
  batch_size: 8  # GPU内存不足可以改为4或2
```

### 2.4 开始训练

```bash
python src/segmentation/train_segmentation.py \
    --config config/room_segmentation.yaml
```

**注意**: 当前版本的训练脚本需要您先实现数据加载部分。

### 2.5 完整训练流程示例

参考 `src/segmentation/train_segmentation.py` 中的TODO部分，补充：

```python
# 加载数据集
with open('data/splits/train.txt') as f:
    train_files = f.read().strip().split('\n')

with open('data/splits/val.txt') as f:
    val_files = f.read().strip().split('\n')

train_dataset = FloorPlanDataset(
    'data/images',
    'data/labels_segmentation/masks',
    train_files,
    transform=get_transforms(config, is_train=True)
)

val_dataset = FloorPlanDataset(
    'data/images',
    'data/labels_segmentation/masks',
    val_files,
    transform=get_transforms(config, is_train=False)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
```

### 2.6 评估分割模型

主要指标:
- **mIoU** (mean Intersection over Union): 主要指标
- **Pixel Accuracy**: 像素准确率

**期望值**:
- mIoU > 0.6: 可用
- mIoU > 0.7: 良好
- mIoU > 0.8: 优秀

### 2.7 测试分割

```bash
python src/segmentation/segment_room.py \
    --image data/images/test_image.jpg \
    --model models/segmentation/best_model.pth \
    --save output/test_segmentation.jpg
```

---

## 第三部分: 训练优化建议

### 3.1 硬件要求

**最低配置**:
- GPU: 6GB显存 (GTX 1060)
- RAM: 16GB
- 存储: 10GB

**推荐配置**:
- GPU: 12GB+ 显存 (RTX 3060+)
- RAM: 32GB
- 存储: 50GB SSD

### 3.2 训练时间估算

基于RTX 3060:
- **目标检测** (YOLOv8s, 100 epochs): 2-4小时
- **语义分割** (U-Net, 100 epochs): 3-6小时

### 3.3 使用预训练模型

强烈推荐使用预训练权重：

```python
# YOLOv8自动下载预训练权重
model = YOLO('yolov8s.pt')  # 会自动下载

# U-Net使用ImageNet预训练encoder
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet'  # 使用ImageNet预训练
)
```

### 3.4 数据增强

已在配置中启用：
- 水平翻转
- 旋转
- 亮度/对比度调整
- Mosaic增强（YOLO）

可根据需要调整参数。

### 3.5 早停机制

为避免过拟合，建议使用早停：

```python
early_stopping_patience = 10  # 10个epoch无提升则停止
```

---

## 第四部分: 常见问题

### Q1: 训练很慢怎么办？

**方案**:
1. 减小图片尺寸 (640 → 512)
2. 减小batch_size
3. 使用更小的模型
4. 使用GPU训练
5. 使用混合精度训练

### Q2: 显存不足

**方案**:
```python
# 减小batch_size
batch_size: 8 → 4 或 2

# 减小图片尺寸
imgsz: 640 → 512 或 416

# 使用gradient accumulation
```

### Q3: 训练loss不下降

**检查**:
1. 学习率是否太大/太小
2. 数据标注是否正确
3. 数据是否有问题
4. 模型是否正确加载

### Q4: 精度不满意

**改进**:
1. 增加数据量
2. 改善标注质量
3. 使用更大的模型
4. 增加训练epoch
5. 调整数据增强
6. 使用集成学习

---

## 第五部分: 模型部署

训练完成后，模型保存在：

```
runs/detect/furniture_detection/weights/
├── best.pt      # 最佳模型
└── last.pt      # 最后一个epoch

models/segmentation/
└── best_model.pth
```

下一步可以：
1. 使用训练好的模型进行推理
2. 导出为ONNX格式加速推理
3. 集成到完整的Agent中

---

## 下一步

继续下一个教程：
- [教程3: 使用Agent进行分析](TUTORIAL_03_使用Agent.md)

---

**上一步**: [← 数据标注教程](TUTORIAL_01_数据标注指南.md)
**下一步**: [使用Agent教程 →](TUTORIAL_03_使用Agent.md)

