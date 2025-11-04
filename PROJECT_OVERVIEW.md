# 平面图智能分析系统 - 项目总览

## 🎯 项目目标

开发一个完整的平面图智能分析系统，实现：
1. **家具图例识别** - 自动识别和统计平面图中的所有家具（门、窗、桌椅等）
2. **房间语义分割** - 将平面图分割成独立的房间区域，区分墙壁和房间
3. **面积计算** - 根据用户输入的比例尺，计算每个房间和总面积

## 📁 项目结构

```
FYP-Floorplan/
│
├── 📂 data/                          # 数据目录
│   ├── images/                       # ✅ 303张原始平面图
│   ├── labels_detection/             # 🔜 待标注：家具检测标签
│   ├── labels_segmentation/          # 🔜 待标注：房间分割标签
│   └── splits/                       # 🔜 数据集划分
│
├── 📂 src/                           # ✅ 源代码（已完成）
│   ├── detection/                    # 家具检测模块
│   │   ├── train_detection.py       # 训练脚本
│   │   └── detect_furniture.py      # 推理脚本
│   ├── segmentation/                 # 房间分割模块
│   │   ├── train_segmentation.py    # 训练脚本
│   │   └── segment_room.py          # 推理脚本
│   ├── utils/                        # 工具模块
│   │   ├── prepare_dataset.py       # 数据集准备
│   │   └── area_calculator.py       # 面积计算
│   └── agent/                        # 主控制器
│       └── floorplan_agent.py       # 完整分析Agent
│
├── 📂 config/                        # ✅ 配置文件（已完成）
│   ├── furniture_detection.yaml     # 检测配置
│   └── room_segmentation.yaml       # 分割配置
│
├── 📂 scripts/                       # ✅ 实用工具（已完成）
│   ├── measure_pixels.py            # 像素测量工具
│   └── visualize_dataset.py         # 数据可视化
│
├── 📂 docs/                          # ✅ 完整文档（已完成）
│   ├── TUTORIAL_01_数据标注指南.md
│   ├── TUTORIAL_02_模型训练.md
│   ├── TUTORIAL_03_使用Agent.md
│   ├── FAQ.md                        # 常见问题
│   ├── 项目架构说明.md
│   └── 实施步骤总结.md
│
├── 📂 models/                        # 🔜 待训练：模型存储
│   ├── detection/                    # 检测模型
│   └── segmentation/                 # 分割模型
│
├── 📄 README.md                      # ✅ 项目说明
├── 📄 QUICKSTART.md                  # ✅ 快速开始
├── 📄 requirements.txt               # ✅ 依赖列表
└── 📄 .gitignore                     # ✅ Git忽略规则
```

## 🚀 快速开始

### 1. 安装环境

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

```bash
# 划分数据集
python src/utils/prepare_dataset.py --images-dir data/images
```

### 3. 开始标注

```bash
# 安装标注工具
pip install labelImg labelme

# 家具检测标注
labelImg data/images data/labels_detection

# 房间分割标注
labelme data/images --output data/labels_segmentation
```

### 4. 训练模型

```bash
# 训练家具检测模型
python src/detection/train_detection.py

# 训练房间分割模型
python src/segmentation/train_segmentation.py
```

### 5. 使用Agent分析

```bash
python src/agent/floorplan_agent.py \
    --image data/images/sample.jpg \
    --scale "1:100" \
    --reference-pixels 200 \
    --reference-length 200
```

## 📚 完整文档索引

### 新手入门
1. **[快速开始](QUICKSTART.md)** - 5分钟了解项目
2. **[教程1：数据标注](docs/TUTORIAL_01_数据标注指南.md)** - 如何标注数据
3. **[教程2：模型训练](docs/TUTORIAL_02_模型训练.md)** - 如何训练模型
4. **[教程3：使用Agent](docs/TUTORIAL_03_使用Agent.md)** - 如何使用系统

### 深入理解
5. **[项目架构说明](docs/项目架构说明.md)** - 技术架构详解
6. **[实施步骤总结](docs/实施步骤总结.md)** - 完整实施指南
7. **[FAQ常见问题](docs/FAQ.md)** - 问题解答

### 完整说明
8. **[README.md](README.md)** - 详细项目说明

## 📊 实施进度

### ✅ 已完成（第1阶段）

- [x] 项目结构搭建
- [x] 核心代码实现
  - [x] 家具检测模块
  - [x] 房间分割模块
  - [x] 面积计算模块
  - [x] Agent集成
- [x] 配置文件
- [x] 工具脚本
- [x] 完整文档
- [x] 教程系列

### 🔜 待完成（第2阶段 - 需要您执行）

- [ ] 数据标注
  - [ ] 标注100-200张检测数据
  - [ ] 标注50-100张分割数据
- [ ] 模型训练
  - [ ] 训练家具检测模型
  - [ ] 训练房间分割模型
- [ ] 测试验证
  - [ ] 测试检测精度
  - [ ] 测试分割效果
  - [ ] 测试面积计算
- [ ] 优化部署
  - [ ] 性能优化
  - [ ] 批量测试

## ⏱️ 时间规划

| 阶段 | 任务 | 预计时间 | 状态 |
|------|------|----------|------|
| 阶段1 | 项目搭建 | 1周 | ✅ 已完成 |
| 阶段2 | 数据标注 | 1-2周 | 🔜 待开始 |
| 阶段3 | 模型训练 | 2-3周 | 🔜 待开始 |
| 阶段4 | 测试优化 | 1-2周 | 🔜 待开始 |
| **总计** | | **5-8周** | **20%已完成** |

## 🎓 技术栈

### 深度学习
- **PyTorch** 2.0+ - 深度学习框架
- **YOLOv8** - 目标检测（家具识别）
- **U-Net** - 语义分割（房间分割）

### 计算机视觉
- **OpenCV** - 图像处理
- **Albumentations** - 数据增强
- **Pillow** - 图像读写

### 工具库
- **NumPy** - 数值计算
- **Pandas** - 数据处理
- **Matplotlib** - 可视化

### 标注工具
- **LabelImg** - 目标检测标注
- **LabelMe** - 语义分割标注

## 💡 核心功能

### 1. 家具检测 (YOLOv8)

```python
from src.detection.detect_furniture import FurnitureDetector

detector = FurnitureDetector('models/best.pt')
results, stats = detector.detect('floorplan.jpg')

# 输出：
# {
#   'total': 23,
#   'by_class': {
#     'door': 3,
#     'window': 5,
#     'bed': 2,
#     ...
#   }
# }
```

### 2. 房间分割 (U-Net)

```python
from src.segmentation.segment_room import RoomSegmenter

segmenter = RoomSegmenter('models/unet.pth')
mask, colored_mask = segmenter.segment('floorplan.jpg')
rooms = segmenter.extract_rooms(mask)

# 输出：4个房间，每个带有面积和位置信息
```

### 3. 面积计算

```python
from src.utils.area_calculator import AreaCalculator

calculator = AreaCalculator()
calculator.set_scale_manual("1:100")
calculator.calibrate_with_reference(200, 200)

area = calculator.calculate_area_from_pixels(50000, unit='m2')
# 输出：12.5 m²
```

### 4. 完整Agent

```python
from src.agent.floorplan_agent import FloorPlanAgent

agent = FloorPlanAgent()
results = agent.analyze(
    image_path='floorplan.jpg',
    scale='1:100',
    reference_pixels=200,
    reference_length=200
)

# 输出：完整分析结果（家具+房间+面积）
```

## 🎯 预期成果

完成整个项目后，您将获得：

### 技术成果
- ✅ 完整的深度学习项目开发经验
- ✅ 目标检测和语义分割实战能力
- ✅ 端到端AI应用开发能力

### 可交付成果
- ✅ 训练好的家具检测模型
- ✅ 训练好的房间分割模型
- ✅ 完整的分析系统
- ✅ 详细的项目文档

### 应用价值
- ✅ 可用于房地产行业户型分析
- ✅ 可用于建筑设计自动化
- ✅ 可用于室内设计辅助
- ✅ 可扩展到其他工程图纸分析

## 📈 性能指标

### 目标性能
| 指标 | 目标值 | 说明 |
|------|--------|------|
| 家具检测 mAP50 | > 0.75 | 可用水平 |
| 房间分割 mIoU | > 0.70 | 可用水平 |
| 面积计算误差 | < 5% | 比例尺校准准确 |
| 单张推理速度 | < 5秒 | GPU环境 |

### 数据需求
- 最少：100-150张标注图片
- 推荐：200-300张标注图片
- 理想：500+张标注图片

## 🔧 系统要求

### 最低配置
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 16GB
- GPU: 6GB显存 (GTX 1060)
- 存储: 10GB 可用空间

### 推荐配置
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 32GB
- GPU: 12GB显存 (RTX 3060/4060)
- 存储: 50GB SSD

## 📞 支持与帮助

### 文档资源
- 📖 [快速开始指南](QUICKSTART.md)
- 📚 [完整教程系列](docs/)
- ❓ [常见问题FAQ](docs/FAQ.md)
- 🏗️ [架构说明](docs/项目架构说明.md)

### 外部资源
- [YOLOv8官方文档](https://docs.ultralytics.com/)
- [PyTorch教程](https://pytorch.org/tutorials/)
- [OpenCV文档](https://docs.opencv.org/)

## 🎉 立即开始

1. **阅读** [QUICKSTART.md](QUICKSTART.md) - 快速了解项目
2. **安装** 依赖环境 - `pip install -r requirements.txt`
3. **学习** [教程1](docs/TUTORIAL_01_数据标注指南.md) - 开始标注数据
4. **训练** 您的第一个模型
5. **测试** 完整的分析流程

## 📝 更新日志

### v1.0.0 (2025-10-30)
- ✅ 完成项目基础架构
- ✅ 实现所有核心功能模块
- ✅ 编写完整文档和教程
- ✅ 提供实用工具脚本

---

**项目状态**: 🟢 框架完成，准备就绪

**下一步**: 开始数据标注 → 详见 [实施步骤总结](docs/实施步骤总结.md)

**祝您开发顺利！如有问题，欢迎查阅文档或提Issue！** 🚀

