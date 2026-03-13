# 如何使用 LabelImg 进行标注

## 📋 准备工作

### 1. 安装 LabelImg

```bash
pip install labelImg
```

### 2. 准备预定义类别文件

LabelImg 需要一个 `predefined_classes.txt` 文件来加载预定义的类别。

文件位置：`data/labels_detection/predefined_classes.txt`

文件内容（17个类别）：
```
door
window
table
chair
bed
sofa
toilet
sink
bathtub
stove
refrigerator
wardrobe
tv
desk
washingmachine
loadbearing_wall
aircondition
```

## 🚀 启动 LabelImg 的正确方法

### 方法1：使用命令行参数（推荐）

```bash
# 基本用法
labelImg [图片目录] [标注保存目录] [预定义类别文件]

# 实际命令
labelImg data/images data/labels_detection data/labels_detection/predefined_classes.txt
```

### 方法2：在 LabelImg 中手动设置

```bash
# 1. 启动 LabelImg
labelImg

# 2. 在 LabelImg 界面中操作：
#    - 点击 "Open Dir" 选择图片目录: data/images
#    - 点击 "Change Save Dir" 选择保存目录: data/labels_detection
#    - 点击 "View" -> "Auto Save mode" 启用自动保存
#    - 点击 "Edit" -> "Label List Panel" 查看类别列表
```

## ⚙️ LabelImg 配置步骤

### 步骤1: 切换到 YOLO 格式

1. 打开 LabelImg
2. 点击左侧的 **"PascalVOC"** 按钮（位于左下方）
3. 按钮会变成 **"YOLO"**
4. 确保显示为 "YOLO" 格式

### 步骤2: 加载预定义类别

LabelImg 会自动从以下位置查找预定义类别文件：

1. **标注保存目录**中的 `predefined_classes.txt`
   - 位置：`data/labels_detection/predefined_classes.txt`
   - ✅ **推荐方式**

2. **当前工作目录**中的 `data/predefined_classes.txt`

3. **LabelImg 安装目录**中的 `data/predefined_classes.txt`

### 步骤3: 验证类别是否加载

标注第一个框时，会弹出类别选择窗口：
- ✅ 如果看到你的17个类别 → 成功！
- ❌ 如果需要手动输入 → 类别文件未加载

## 🎯 完整标注流程

### 使用原始101张图片标注（推荐）

```bash
# 1. 提取原始图片到单独目录
python scripts/extract_original_images.py

# 2. 使用 LabelImg 标注原始图片
labelImg data/images_original data/labels_detection/predefined_classes.txt data/labels_detection

# 3. 在 LabelImg 中：
#    - 按 W 创建标注框
#    - 从列表中选择类别（不用手动输入）
#    - 按 Ctrl+S 保存
#    - 按 D 下一张图片

# 4. 标注完成后，自动生成其他202张的标注
python scripts/auto_generate_labels.py
```

### 直接标注全部303张（备选）

```bash
labelImg data/images data/labels_detection data/labels_detection/predefined_classes.txt
```

## 🔧 LabelImg 快捷键

| 快捷键 | 功能 |
|--------|------|
| `W` | 创建矩形标注框 |
| `D` | 下一张图片 |
| `A` | 上一张图片 |
| `Del` | 删除选中的标注框 |
| `Ctrl+S` | 保存当前标注 |
| `Ctrl+D` | 复制当前标注框 |
| `Space` | 标记当前图片为已验证 |
| `Ctrl+U` | 从图片列表中选择图片 |
| `Ctrl++` | 放大 |
| `Ctrl+-` | 缩小 |
| `↑↓←→` | 移动选中的标注框 |

## 📝 标注最佳实践

### 1. 标注准确性

```
✅ 好的标注：
┌──────────┐
│  [门]    │  ← 框紧贴门的边缘
└──────────┘

❌ 差的标注：
┌─────────────────┐
│    [门]         │  ← 框太大，包含了墙壁
└─────────────────┘
```

### 2. 标注完整性

- ✅ 标注所有可见的家具图例
- ✅ 即使部分遮挡也要标注
- ❌ 不要遗漏小的图例

### 3. 类别一致性

- ✅ 相同的物体使用相同的类别
- ✅ 不确定的类别可以跳过
- ❌ 不要混淆相似的类别（如 sink 和 washbasin）

### 4. 避免重复标注

- ✅ 一个物体只标注一次
- ❌ 不要重复框选同一个物体

## 🐛 常见问题解决

### Q1: LabelImg 没有显示预定义类别？

**解决方案：**

1. 确认 `predefined_classes.txt` 在正确位置
2. 文件内容每行一个类别，不要有空行
3. 重启 LabelImg

```bash
# 确认文件存在
ls data/labels_detection/predefined_classes.txt

# 查看文件内容
cat data/labels_detection/predefined_classes.txt
```

### Q2: 标注保存为 XML 格式而不是 TXT？

**解决方案：**
- 点击左下角的 "PascalVOC" 按钮切换为 "YOLO" 格式

### Q3: 类别选择窗口没有出现？

**解决方案：**
- 右键点击标注框
- 选择 "Edit Label"
- 或者在左侧标注列表中双击标注框

### Q4: 标注框保存后找不到？

**解决方案：**
```bash
# 检查保存目录
ls data/labels_detection/

# 查看某个标注文件
cat data/labels_detection/FloorPlan-1-*.txt
```

### Q5: LabelImg 启动失败？

**解决方案：**
```bash
# 重新安装
pip uninstall labelImg
pip install labelImg

# 或使用 labelme（备选工具）
pip install labelme
```

## 📂 文件结构

标注完成后，你应该有：

```
data/
├── images/                          # 303张图片
│   ├── FloorPlan-1-xxx.jpg
│   ├── FloorPlan-2-xxx.jpg
│   └── ...
│
├── images_original/                 # 101张原始图片（可选）
│   ├── FloorPlan-1-xxx.jpg
│   ├── FloorPlan-2-xxx.jpg
│   └── ...
│
└── labels_detection/
    ├── predefined_classes.txt       # 预定义类别文件
    ├── classes.txt                  # YOLO格式类别文件
    ├── FloorPlan-1-xxx.txt          # 标注文件
    ├── FloorPlan-2-xxx.txt
    └── ...
```

## 🎯 标注进度追踪

### 创建进度记录

```bash
# 统计已标注的文件数
ls data/labels_detection/*.txt | wc -l

# 查看标注文件列表
ls data/labels_detection/*.txt > 标注进度.txt
```

### 查找未标注的图片

```python
# 创建检查脚本
python -c "
import os
images = set([os.path.splitext(f)[0] for f in os.listdir('data/images')])
labels = set([os.path.splitext(f)[0] for f in os.listdir('data/labels_detection') if f.endswith('.txt') and f != 'classes.txt'])
missing = images - labels
print(f'未标注的图片: {len(missing)}张')
if missing:
    for img in sorted(missing)[:10]:
        print(f'  - {img}')
"
```

## 🚀 下一步

标注完成后：

1. **验证标注质量**
   ```bash
   python scripts/visualize_dataset.py
   ```

2. **生成增强标注**（如果只标注了101张）
   ```bash
   python scripts/auto_generate_labels.py
   ```

3. **准备训练数据**
   ```bash
   python src/utils/prepare_dataset.py
   ```

4. **开始训练模型**
   ```bash
   python src/detection/train_detection.py
   ```

---

## 📌 快速参考命令

```bash
# 标注原始图片（推荐）
labelImg data/images_original data/labels_detection data/labels_detection/predefined_classes.txt

# 标注全部图片
labelImg data/images data/labels_detection data/labels_detection/predefined_classes.txt

# 查看标注数量
ls data/labels_detection/*.txt | grep -v classes.txt | wc -l

# 可视化验证
python scripts/visualize_dataset.py
```

---

**祝标注顺利！如有问题，随时查阅本文档。** 🎉



