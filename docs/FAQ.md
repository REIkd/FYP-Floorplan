# 常见问题解答 (FAQ)

## 项目相关

### Q1: 这个项目适合初学者吗？

**A**: 需要一定基础。建议您具备：
- Python编程基础
- 基本的深度学习概念
- 了解目标检测和语义分割

如果是初学者，建议先学习：
1. [深度学习入门课程](https://www.deeplearning.ai/)
2. [YOLOv8官方教程](https://docs.ultralytics.com/)
3. [语义分割入门](https://www.pyimagesearch.com/)

### Q2: 需要多少数据才能训练好模型？

**A**: 
- **最少**: 50-100张标注图片（可以开始实验）
- **推荐**: 150-200张（获得不错的效果）
- **理想**: 300+张（获得最佳效果）

可以使用数据增强来扩充数据集。

### Q3: 训练需要多长时间？

**A**: 基于RTX 3060 GPU:
- 家具检测 (YOLOv8s, 100 epochs): 2-4小时
- 房间分割 (U-Net, 100 epochs): 3-6小时

如果使用CPU训练会慢10-20倍。

### Q4: 没有GPU可以训练吗？

**A**: 可以，但会非常慢。建议：
1. 使用Google Colab（免费GPU）
2. 使用云服务器（AWS, Azure, 阿里云等）
3. 减小模型和数据集规模
4. 考虑使用预训练模型微调

---

## 数据标注

### Q5: 标注工具推荐哪个？

**A**: 
- **目标检测**: LabelImg（离线）或 Roboflow（在线）
- **语义分割**: LabelMe（离线）或 CVAT（在线）

推荐新手使用在线工具，界面友好且有教程。

### Q6: 标注太耗时怎么办？

**A**: 几个建议：
1. 先标注一小部分（50张），训练初步模型
2. 使用模型预测未标注图片
3. 人工校正预测结果（比从头标注快很多）
4. 多人协作标注
5. 考虑外包标注工作

### Q7: 标注时某些图例看不清怎么办？

**A**: 
1. 跳过不确定的物体
2. 创建"unknown"类别
3. 预处理图像（增强对比度、去噪等）
4. 参考相似的已标注图片

### Q8: 需要标注所有家具吗？

**A**: 不一定。可以：
1. 只标注主要家具（门、窗、床、桌子）
2. 根据应用需求选择类别
3. 后续可以增量添加新类别

---

## 模型训练

### Q9: 训练时显存不足怎么办？

**A**: 几个解决方案：
```yaml
# 1. 减小batch size
batch_size: 16 → 8 或 4

# 2. 减小图片尺寸
imgsz: 640 → 512 或 416

# 3. 使用更小的模型
model_size: 's' → 'n' (nano)

# 4. 使用混合精度训练
# 在代码中添加: torch.cuda.amp
```

### Q10: 训练loss不下降怎么办？

**A**: 检查：
1. **学习率**: 太大或太小都不行，尝试 1e-3 到 1e-5
2. **数据**: 检查标注是否正确
3. **模型**: 确认模型结构和参数正确
4. **预处理**: 检查数据增强是否过度

可以尝试：
```python
# 降低学习率
lr0: 0.001 → 0.0001

# 调整优化器
optimizer: AdamW → Adam

# 减少数据增强
mosaic: 1.0 → 0.5
```

### Q11: 如何知道模型训练好了？

**A**: 观察指标：

**目标检测**:
- mAP50 > 0.7: 可用
- mAP50 > 0.8: 良好
- mAP50 > 0.9: 优秀

**语义分割**:
- mIoU > 0.6: 可用
- mIoU > 0.7: 良好
- mIoU > 0.8: 优秀

同时检查验证集loss是否收敛。

### Q12: 模型过拟合怎么办？

**A**: 
1. 增加数据量
2. 增强数据增强
3. 使用更小的模型
4. 添加正则化（dropout, weight decay）
5. 使用早停机制

```python
# 增加weight decay
weight_decay: 0.0005 → 0.001

# 启用早停
early_stopping_patience: 10
```

---

## 使用和推理

### Q13: 比例尺校准怎么做？

**A**: 详细步骤：

1. 在平面图上找一条已知长度的线段
2. 使用我们提供的工具测量像素长度：
   ```bash
   python scripts/measure_pixels.py --image your_image.jpg
   ```
3. 在工具中点击线段的两端
4. 记录像素长度（如200像素）
5. 使用时提供参数：
   ```bash
   --reference-pixels 200 --reference-length 实际长度(cm)
   ```

### Q14: 检测结果不准确怎么办？

**A**: 
1. 调整置信度阈值：
   ```bash
   --conf 0.25  # 默认值，可以调高（0.3-0.5）
   ```
2. 检查模型是否训练充分
3. 测试图片是否与训练集相似
4. 考虑重新训练或增加训练数据

### Q15: 计算的面积不对？

**A**: 最常见原因是校准不准确：
1. 确保参考线段足够长（减少误差）
2. 多测量几次取平均
3. 检查比例尺是否正确
4. 验证像素测量是否准确

### Q16: 可以批量处理吗？

**A**: 可以！参考教程3中的批量处理脚本：
```bash
python scripts/batch_analyze.py
```

或使用循环：
```bash
for file in data/images/*.jpg; do
    python src/agent/floorplan_agent.py --image "$file" ...
done
```

---

## 技术问题

### Q17: 支持哪些图片格式？

**A**: 支持常见格式：
- JPG/JPEG ✓
- PNG ✓
- BMP ✓
- TIFF ✓

### Q18: 可以用于其他类型的图纸吗？

**A**: 可以！稍作调整即可用于：
- 建筑平面图 ✓
- 电气布线图 ✓
- 管道图 ✓
- 其他工程图纸 ✓

需要重新标注和训练对应类别。

### Q19: 如何部署到生产环境？

**A**: 几种方式：

**1. Web API**:
```python
# 使用FastAPI
uvicorn webapp:app --host 0.0.0.0 --port 8000
```

**2. Docker容器**:
```dockerfile
FROM pytorch/pytorch:latest
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "src/agent/floorplan_agent.py"]
```

**3. ONNX导出**（加速推理）:
```python
model.export(format='onnx')
```

### Q20: 可以在移动端运行吗？

**A**: 可以，但需要：
1. 导出为轻量级格式（ONNX, TFLite）
2. 使用量化减小模型大小
3. 使用移动端推理框架（如 NCNN, MNN）

或者采用服务端推理+移动端展示的架构。

---

## 项目扩展

### Q21: 如何添加新的家具类别？

**A**: 
1. 标注新类别的数据
2. 更新配置文件中的类别列表
3. 重新训练模型

或者：
1. 只标注新类别的数据
2. 从之前的模型继续训练（迁移学习）

### Q22: 可以识别房间名称吗？

**A**: 本项目不包含OCR功能，但可以集成：
```python
import easyocr
reader = easyocr.Reader(['ch_sim', 'en'])
results = reader.readtext(image)
```

提取文字后可以匹配到对应房间。

### Q23: 可以生成3D模型吗？

**A**: 本项目专注2D分析，但可以扩展：
1. 使用分割结果提取墙壁线条
2. 使用算法重建3D模型
3. 参考相关论文和开源项目

推荐工具：
- Blender + Python API
- Three.js (Web 3D)
- Unity/Unreal (游戏引擎)

### Q24: 可以自动生成户型报告吗？

**A**: 完全可以！例如：
```python
def generate_report(results):
    """生成户型评估报告"""
    report = {
        '总面积': results['area']['total_area'],
        '房间数': len(results['rooms']),
        '采光': evaluate_lighting(results['furniture']),
        '户型方正度': calculate_regularity(results['rooms']),
        '动静分区': check_zoning(results)
    }
    return report
```

可以基于规则或机器学习生成评分和建议。

---

## 性能优化

### Q25: 如何加速推理？

**A**: 
1. **使用GPU**: 显著提速
2. **导出ONNX**: 减少开销
3. **TensorRT**: Nvidia GPU专用优化
4. **量化**: INT8量化可加速2-4倍
5. **批处理**: 一次处理多张图片

```python
# 批处理示例
results = model.predict(
    [img1, img2, img3],  # 多张图片
    batch=3
)
```

### Q26: 模型文件太大怎么办？

**A**: 
1. **使用更小的模型**: YOLOv8n 而不是 YOLOv8x
2. **模型剪枝**: 去除不重要的权重
3. **知识蒸馏**: 用大模型训练小模型
4. **量化**: 减小模型精度（FP16 或 INT8）

---

## 其他

### Q27: 商业使用需要注意什么？

**A**: 
- 本项目代码可自由使用
- 注意第三方库的许可证
- YOLOv8: AGPL-3.0（商业使用需购买许可）
- 建议咨询法律顾问

### Q28: 在哪里获取帮助？

**A**: 
1. 查看本文档和教程
2. 提交GitHub Issue
3. 查看代码注释
4. 参考官方文档（YOLO, PyTorch等）

### Q29: 如何贡献代码？

**A**: 
1. Fork项目
2. 创建feature分支
3. 提交Pull Request
4. 说明改动内容和目的

欢迎贡献！

### Q30: 项目后续计划？

**A**: 可能的方向：
- 支持多楼层分析
- 3D重建
- 移动端应用
- 自动户型评分
- 数据集公开

---

## 还有问题？

如果您的问题没有被列出，请：
1. 提交Issue: [GitHub Issues]
2. 查看相关文档和教程
3. 参考代码注释

**祝您使用愉快！** 🎉

