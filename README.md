# 建筑平面图分析系统

基于深度学习技术的建筑平面图识别和分析系统，能够自动识别平面图中的对象并计算尺寸。

## 功能特性

- 🏗️ **对象识别**: 自动识别平面图中的门、窗、楼梯、电梯等建筑元素
- 📏 **尺寸计算**: 基于比例尺计算房间面积和对象尺寸
- 📊 **统计分析**: 统计各类对象的数量和分布
- 🖥️ **用户界面**: 友好的Web界面，支持图片上传和分析
- 🔍 **数据验证**: 完整的数据集验证和测试功能

## 项目结构

```
FYP-Floorplan/
├── app/                    # Flask Web应用
├── models/                 # 深度学习模型
├── data/                   # 数据集和预处理
├── utils/                  # 工具函数
├── tests/                  # 测试文件
└── requirements.txt        # 依赖包
```

## 安装和使用

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 运行应用:
```bash
python app.py
```

3. 访问 http://localhost:5000 使用Web界面

## 数据集要求

- 平面图图像格式: JPG, PNG
- 包含比例尺信息
- 标注文件: YOLO格式
- 对象类别: 门、窗、楼梯、电梯、房间等

## 技术栈

- **深度学习**: PyTorch, YOLOv8
- **图像处理**: OpenCV, PIL
- **Web框架**: Flask
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib, Seaborn
