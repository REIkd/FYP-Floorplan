# 平面图智能分析系统 (Floor Plan Analysis Agent)

## 项目概述
本项目旨在开发一个智能平面图分析系统，可以：
1. **家具图例识别与统计** - 识别并统计平面图中的所有家具图例（门、桌子、椅子等）
2. **房间语义分割** - 将平面图分割成房间和墙壁
3. **面积计算** - 根据用户输入的比例尺计算房间总面积

## 数据集信息
- 位置: `data/images/` 
- 数量: 303张平面图图片
- 格式: JPG

## 项目实施步骤

### 第一阶段：环境搭建与数据准备 (1-2周)

#### 1.1 安装依赖环境
```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch / TensorFlow (深度学习框架)
- Ultralytics YOLOv8 (目标检测)
- MMSegmentation (语义分割)
- OpenCV (图像处理)
- LabelImg / LabelMe (数据标注)

#### 1.2 数据标注

**目标检测标注** (家具图例识别):
- 工具: LabelImg (YOLO格式) 或 Roboflow
- 需要标注的类别示例:
  - door (门)
  - window (窗)
  - table (桌子)
  - chair (椅子)
  - bed (床)
  - sofa (沙发)
  - toilet (马桶)
  - sink (水槽)
  等等...
  
- 标注格式: YOLO txt格式 (class_id, x_center, y_center, width, height)
- 建议标注数量: 至少150-200张图片作为训练集

**语义分割标注** (房间分割):
- 工具: LabelMe 或 CVAT
- 需要标注的类别:
  - background (背景)
  - wall (墙壁)
  - room (房间区域)
  - door_area (门区域)
  - window_area (窗户区域)
  
- 标注格式: 像素级别的mask (PNG格式)
- 建议标注数量: 至少100-150张图片

### 第二阶段：模型训练 (2-3周)

#### 2.1 家具检测模型训练
使用 YOLOv8 进行目标检测训练：
```bash
python train_detection.py --data config/furniture_detection.yaml --epochs 100
```

#### 2.2 房间分割模型训练
使用 U-Net 或 DeepLabV3+ 进行语义分割训练：
```bash
python train_segmentation.py --config config/room_segmentation.yaml --epochs 100
```

### 第三阶段：推理与分析 (1周)

#### 3.1 家具统计
```bash
python analyze_furniture.py --image path/to/floorplan.jpg
```

输出示例:
```
检测结果:
- door: 3个
- window: 5个
- table: 2个
- chair: 8个
```

#### 3.2 房间分割与面积计算
```bash
python analyze_room.py --image path/to/floorplan.jpg --scale 1:100
```

输出示例:
```
房间分割结果:
- 房间总数: 4个
- 房间1面积: 25.6 m²
- 房间2面积: 18.3 m²
- 房间3面积: 12.5 m²
- 房间4面积: 8.7 m²
- 总面积: 65.1 m²
```

### 第四阶段：Agent集成 (1-2周)

创建统一的分析Agent，整合所有功能：
```bash
python floorplan_agent.py --image path/to/floorplan.jpg --scale 1:100
```

## 大模型 (LLM) 与知识库集成 (Ollama + RAG)

为提升 Agent 的智能交互能力和对平面图设计规范的理解，系统集成了基于 Ollama 的本地大语言模型与向量数据库（Vector Database），实现检索增强生成（RAG）和模型训练微调。

### 1. Ollama API 集成与使用
Ollama 允许在本地轻量化运行开源大语言模型（如 Qwen, Llama 3 等），既能保证图纸数据的隐私安全，又能提供强大的自然语言理解能力。

**安装与运行模型**:
```bash
# 启动本地 Ollama 服务并运行模型 (例如 qwen2.5)
ollama run qwen2.5:7b
```

**Python 中调用 Ollama API**:
使用官方 `ollama` 库与本地服务进行交互：
```python
import ollama

# 传入平面图分析结果供模型解读
prompt = "基于以下检测数据生成一份户型评估报告：房间总面积 85平米，门3个，窗户4个，卧室面积仅7平米..."
response = ollama.chat(model='qwen2.5:7b', messages=[
  {'role': 'system', 'content': '你是一个专业的室内设计师和架构分析师。'},
  {'role': 'user', 'content': prompt}
])
print(response['message']['content'])
```

### 2. 数据库存储与 RAG (检索增强生成)
为了让 LLM 具备专业的建筑知识（如《住宅设计规范》、优秀户型设计案例），我们使用向量数据库来存储这些知识，并在用户提问时进行检索匹配，从而辅助 LLM 生成更专业、准确的建议。

**技术栈推荐**:
- **向量数据库**：ChromaDB (推荐本地轻量化使用), Qdrant 或 Milvus
- **框架**：LangChain 或 LlamaIndex

**实现流程**:
1. **数据准备与存储**：
   将建筑规范文档转化为文本块，使用 Embedding 模型（如 `nomic-embed-text`）转化为向量并存入数据库。
   ```python
   import chromadb

   # 初始化本地向量数据库
   chroma_client = chromadb.PersistentClient(path="./rag_db")
   collection = chroma_client.get_or_create_collection(name="architecture_rules")
   
   # 将建筑规范存入数据库
   documents = ["双人卧室面积不应小于 9 平方米，单人卧室不应小于 5 平方米。", "卫生间应配备通风口。"]
   collection.add(
       documents=documents,
       ids=["rule_1", "rule_2"]
   )
   ```

2. **检索与问答 (RAG)**：
   当用户或系统询问平面图设计的合理性时，先从数据库中检索相关的规范条文，然后将其作为上下文喂给 Ollama。
   ```python
   # 分析过程中的提问
   query = "这个平面图中有一个只有7平米的双人卧室，是否符合规范？"
   
   # 检索最相关的规范
   results = collection.query(query_texts=[query], n_results=1)
   context = results['documents'][0][0]  # 提取相关上下文
   
   # 结合上下文请求 Ollama
   rag_prompt = f"参考规范：{context}\n请基于规范回答：{query}"
   response = ollama.generate(model='qwen2.5:7b', prompt=rag_prompt)
   print(response['response'])
   ```

### 3. 本地数据存储与模型微调 (Fine-Tuning)
如果基础模型的表现仍有不足，可以利用常规关系型数据库（如 SQLite/PostgreSQL）长期收集数据进行 LLM 微调：
1. **数据收集**：在数据库中记录系统的**平面图结构化输出**（如房间面积数组、家具列表）以及人工修改/确认后的**优质评估报告**。
2. **构建数据集**：将积累的数据导出为 `JSONL` 格式的指令微调数据集（Instruction Tuning Dataset）。
3. **模型训练**：使用 LLaMA-Factory 或 Unsloth 等微调框架，对 Ollama 本地的模型进行 LoRA 微调。
4. **模型部署**：将微调后导出的 GGUF 模型文件重新导入进 Ollama 供项目使用，使其成为专属于户型分析的垂类大模型。

## 项目结构
```
FYP-Floorplan/
├── data/
│   ├── images/              # 原始平面图图片
│   ├── labels_detection/    # 目标检测标注
│   ├── labels_segmentation/ # 语义分割标注
│   └── splits/              # 训练/验证/测试集划分
├── models/
│   ├── detection/           # 家具检测模型
│   └── segmentation/        # 房间分割模型
├── src/
│   ├── detection/           # 检测相关代码
│   ├── segmentation/        # 分割相关代码
│   ├── utils/               # 工具函数
│   └── agent/               # 分析Agent
├── config/                  # 配置文件
├── notebooks/               # Jupyter notebooks (数据探索)
├── results/                 # 结果输出
├── requirements.txt         # 依赖列表
└── README.md
```

## 技术栈选择建议

### 方案A: YOLOv8 + U-Net (推荐初学者)
- **优点**: 简单易用，文档丰富，训练速度快
- **检测**: YOLOv8 (Ultralytics)
- **分割**: U-Net (PyTorch)

### 方案B: Mask R-CNN (实例分割一体化)
- **优点**: 同时完成检测和分割
- **缺点**: 训练较慢，需要更多数据
- **框架**: Detectron2 (Facebook AI)

### 方案C: Transformer架构 (最新技术)
- **检测**: DETR (Detection Transformer)
- **分割**: SegFormer
- **优点**: 性能最好
- **缺点**: 需要更多计算资源

## 关键技术点

### 1. 比例尺识别
- 使用OCR识别平面图上的比例尺文字
- 或让用户手动输入比例尺 (如 1:100)

### 2. 像素到实际面积转换
```python
# 假设比例尺为 1:100 (图上1cm = 实际100cm)
scale_ratio = 100
pixel_per_cm = image_dpi / 2.54  # DPI转换
real_area_m2 = (pixel_area / (pixel_per_cm ** 2)) * (scale_ratio ** 2) / 10000
```

### 3. 后处理优化
- 非极大值抑制 (NMS) 去除重复检测
- 形态学操作优化分割结果
- 连通域分析统计房间数量

## 评估指标

### 检测模型
- mAP (mean Average Precision)
- Precision, Recall
- F1-Score

### 分割模型
- mIoU (mean Intersection over Union)
- Pixel Accuracy
- Dice Coefficient

## 未来扩展

1. **3D重建**: 从2D平面图生成3D模型
2. **户型评分**: 基于AI评估户型合理性
3. **自动设计建议**: 给出家具摆放建议
4. **多楼层分析**: 支持多层平面图分析

## 参考资源

- [YOLOv8 文档](https://docs.ultralytics.com/)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [CubiCasa5K Dataset](https://github.com/CubiCasa/CubiCasa5k) - 平面图数据集参考
- [RPLAN Dataset](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) - 另一个平面图数据集

## 时间线估计

- 第1-2周: 环境搭建 + 数据标注
- 第3-4周: 目标检测模型训练与调优
- 第5-6周: 语义分割模型训练与调优
- 第7周: 面积计算与后处理
- 第8周: Agent集成与测试
- 第9-10周: 优化与文档

**总计: 约2-3个月完成完整系统**

