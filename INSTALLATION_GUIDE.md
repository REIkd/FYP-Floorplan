# 建筑平面图分析系统 - 安装指南

## 系统要求

- Python 3.8 或更高版本
- 8GB RAM (推荐 16GB)
- 2GB 可用磁盘空间
- 支持的操作系统: Windows, macOS, Linux

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository-url>
cd FYP-Floorplan
```

### 2. 创建虚拟环境 (推荐)
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. 安装依赖
```bash
# 安装所有依赖包
pip install -r requirements.txt

# 或者手动安装主要依赖
pip install torch torchvision opencv-python numpy pillow matplotlib flask flask-cors
```

### 4. 验证安装
```bash
# 检查依赖是否正确安装
python start.py --test

# 运行演示
python demo.py
```

## 快速开始

### 1. 启动系统
```bash
python start.py
```

### 2. 访问Web界面
打开浏览器访问: http://localhost:5000

### 3. 上传平面图
- 点击上传区域或拖拽图片文件
- 设置比例尺 (默认1:100)
- 点击分析按钮

### 4. 查看结果
- 检测到的对象列表
- 尺寸计算结果
- 统计信息

## 故障排除

### 常见问题

#### 1. 依赖安装失败
```bash
# 升级pip
pip install --upgrade pip

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 2. 模型文件缺失
系统会自动使用模拟检测模式，不影响基本功能演示。

#### 3. 端口被占用
```bash
# 修改端口
python app.py --port 5001
```

#### 4. 内存不足
- 关闭其他应用程序
- 使用较小的图像文件
- 调整检测参数

### 性能优化

#### 1. GPU加速 (可选)
```bash
# 安装CUDA版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. 内存优化
- 使用较小的图像尺寸
- 调整批处理大小
- 启用内存映射

## 开发环境设置

### 1. 代码格式化
```bash
pip install black flake8
black .
flake8 .
```

### 2. 运行测试
```bash
# 运行所有测试
python run_tests.py

# 运行特定测试
python run_tests.py --module models
python run_tests.py --module utils
python run_tests.py --module integration
```

### 3. 调试模式
```bash
# 启用调试模式
export FLASK_ENV=development
python app.py
```

## 部署指南

### 1. 生产环境配置
```bash
# 设置环境变量
export FLASK_ENV=production
export SECRET_KEY=your-secret-key

# 启动应用
python app.py
```

### 2. Docker部署 (可选)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

### 3. 云部署
- 支持Heroku, AWS, Azure等云平台
- 配置环境变量
- 设置静态文件服务

## 使用示例

### 1. 基本使用
```python
from models.floorplan_detector import FloorplanDetector
from utils.image_processor import ImageProcessor
from utils.scale_calculator import ScaleCalculator

# 初始化组件
detector = FloorplanDetector()
processor = ImageProcessor()
calculator = ScaleCalculator()

# 处理图像
image = cv2.imread('floorplan.jpg')
processed = processor.preprocess_array(image)
detections = detector.detect_objects(processed)
calculations = calculator.calculate_sizes(detections, 100)
```

### 2. 批量处理
```python
import os
from pathlib import Path

# 批量处理文件夹中的图像
input_dir = Path('input_images')
output_dir = Path('results')

for image_file in input_dir.glob('*.jpg'):
    # 处理每个图像
    image = cv2.imread(str(image_file))
    detections = detector.detect_objects(image)
    calculations = calculator.calculate_sizes(detections, 100)
    
    # 保存结果
    result_file = output_dir / f"{image_file.stem}_result.json"
    with open(result_file, 'w') as f:
        json.dump(calculations, f, indent=2)
```

### 3. API使用
```python
import requests

# 上传图像进行分析
with open('floorplan.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/upload', files=files)
    
# 获取分析结果
data = response.json()
analysis_response = requests.post('http://localhost:5000/analyze', 
                                json={'filename': data['filename'], 'scale_ratio': 100})
```

## 技术支持

### 1. 文档资源
- 项目文档: README.md
- API文档: 查看代码注释
- 示例代码: demo.py

### 2. 问题报告
- 检查日志文件: logs/app.log
- 运行诊断: python start.py --test
- 查看错误信息: 控制台输出

### 3. 社区支持
- GitHub Issues: 报告bug和功能请求
- 技术讨论: 查看项目讨论区
- 贡献代码: 提交Pull Request

## 更新日志

### v1.0.0 (2024-10-23)
- ✅ 初始版本发布
- ✅ 基础功能实现
- ✅ Web界面完成
- ✅ 测试覆盖完成

### 未来计划
- 🔄 模型优化
- 🔄 性能提升
- 🔄 功能扩展
- 🔄 用户体验改进

---

**安装完成后，运行 `python start.py` 启动系统！**
