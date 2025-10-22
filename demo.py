"""
演示脚本
展示建筑平面图分析系统的功能
"""

import numpy as np
import cv2
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from models.floorplan_detector import FloorplanDetector
from utils.image_processor import ImageProcessor
from utils.scale_calculator import ScaleCalculator
from utils.data_validator import DataValidator

def create_demo_image():
    """创建演示图像"""
    print("🎨 创建演示平面图...")
    
    # 创建一个简单的平面图
    img = np.ones((800, 1000, 3), dtype=np.uint8) * 255  # 白色背景
    
    # 绘制外墙
    cv2.rectangle(img, (50, 50), (950, 750), (0, 0, 0), 3)
    
    # 绘制内墙
    cv2.line(img, (500, 50), (500, 400), (0, 0, 0), 2)  # 垂直墙
    cv2.line(img, (50, 400), (500, 400), (0, 0, 0), 2)  # 水平墙
    cv2.line(img, (500, 400), (950, 400), (0, 0, 0), 2)  # 水平墙
    
    # 绘制门
    cv2.rectangle(img, (450, 50), (550, 100), (0, 255, 0), -1)  # 门1
    cv2.rectangle(img, (200, 350), (250, 400), (0, 255, 0), -1)  # 门2
    
    # 绘制窗户
    cv2.rectangle(img, (100, 50), (150, 100), (255, 0, 0), -1)  # 窗1
    cv2.rectangle(img, (800, 50), (850, 100), (255, 0, 0), -1)  # 窗2
    
    # 绘制楼梯
    cv2.rectangle(img, (600, 500), (700, 600), (0, 0, 255), -1)  # 楼梯
    
    # 绘制房间标签
    cv2.putText(img, "Living Room", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Bedroom", (600, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Kitchen", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # 绘制比例尺
    cv2.line(img, (50, 800), (150, 800), (0, 0, 0), 3)
    cv2.putText(img, "1:100", (50, 850), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img

def demo_image_processing():
    """演示图像处理功能"""
    print("\n🖼️  演示图像处理功能...")
    
    # 创建演示图像
    demo_img = create_demo_image()
    
    # 保存演示图像
    cv2.imwrite('demo_floorplan.jpg', demo_img)
    print("✅ 演示图像已保存为 demo_floorplan.jpg")
    
    # 初始化图像处理器
    processor = ImageProcessor()
    
    # 预处理图像
    processed_img = processor.preprocess_array(demo_img)
    print(f"✅ 图像预处理完成，尺寸: {processed_img.shape}")
    
    # 图像增强
    enhanced_img = processor.enhance_image(demo_img, 'auto')
    print("✅ 图像增强完成")
    
    # 获取图像信息
    img_info = processor.get_image_info(demo_img)
    print(f"✅ 图像信息: {img_info['width']}x{img_info['height']}, {img_info['channels']}通道")
    
    return demo_img

def demo_object_detection():
    """演示对象检测功能"""
    print("\n🔍 演示对象检测功能...")
    
    # 创建演示图像
    demo_img = create_demo_image()
    
    # 初始化检测器
    detector = FloorplanDetector()
    
    # 检测对象
    detections = detector.detect_objects(demo_img)
    print(f"✅ 检测到 {len(detections)} 个对象")
    
    # 显示检测结果
    for i, detection in enumerate(detections):
        print(f"  对象 {i+1}: {detection['class']} (置信度: {detection['confidence']:.2f})")
    
    # 获取检测统计
    stats = detector.get_detection_statistics(detections)
    print(f"✅ 检测统计: {stats['total_objects']} 个对象, 平均置信度: {stats['average_confidence']:.2f}")
    
    return detections

def demo_size_calculation():
    """演示尺寸计算功能"""
    print("\n📏 演示尺寸计算功能...")
    
    # 创建模拟检测结果
    mock_detections = [
        {
            'class': 'door',
            'width': 50,
            'height': 100,
            'area': 5000,
            'confidence': 0.8
        },
        {
            'class': 'window',
            'width': 80,
            'height': 60,
            'area': 4800,
            'confidence': 0.9
        },
        {
            'class': 'room',
            'width': 400,
            'height': 300,
            'area': 120000,
            'confidence': 0.95
        }
    ]
    
    # 初始化计算器
    calculator = ScaleCalculator()
    
    # 计算尺寸
    calculations = calculator.calculate_sizes(mock_detections, 100)
    print(f"✅ 完成 {len(calculations)} 个对象的尺寸计算")
    
    # 显示计算结果
    for calc in calculations:
        real_dims = calc['real_dimensions']
        print(f"  {calc['type']}: {real_dims['width']:.2f}m x {real_dims['height']:.2f}m, 面积: {real_dims['area']:.2f}m²")
    
    # 计算总面积
    total_area = calculator.calculate_total_area(mock_detections, 'room')
    print(f"✅ 房间总面积: {total_area:.2f}m²")
    
    # 统计对象数量
    counts = calculator.calculate_object_counts(mock_detections)
    print(f"✅ 对象统计: {counts}")
    
    return calculations

def demo_data_validation():
    """演示数据验证功能"""
    print("\n✅ 演示数据验证功能...")
    
    # 创建测试数据集
    test_data_dir = Path('test_dataset')
    test_data_dir.mkdir(exist_ok=True)
    
    # 创建测试图像
    test_img = create_demo_image()
    cv2.imwrite(str(test_data_dir / 'test.jpg'), test_img)
    
    # 创建测试标注文件
    with open(test_data_dir / 'test.txt', 'w') as f:
        f.write("0 0.5 0.5 0.1 0.1\n")  # 门
        f.write("1 0.3 0.3 0.2 0.15\n")  # 窗
        f.write("4 0.2 0.2 0.4 0.3\n")   # 房间
    
    # 初始化验证器
    validator = DataValidator()
    
    # 验证数据集
    validation_results = validator.validate_dataset(str(test_data_dir))
    print(f"✅ 数据集验证完成: {validation_results['valid_files']}/{validation_results['total_files']} 文件有效")
    
    # 验证检测结果
    mock_detections = [
        {'class': 'door', 'confidence': 0.8, 'bbox': [100, 100, 50, 100]},
        {'class': 'window', 'confidence': 0.9, 'bbox': [200, 150, 80, 60]}
    ]
    
    detection_validation = validator.validate_detection_results(mock_detections)
    print(f"✅ 检测结果验证: {detection_validation['valid_detections']}/{detection_validation['total_detections']} 有效")
    
    # 清理测试数据
    import shutil
    shutil.rmtree(test_data_dir, ignore_errors=True)
    
    return validation_results

def demo_full_pipeline():
    """演示完整处理流程"""
    print("\n🔄 演示完整处理流程...")
    
    try:
        # 1. 创建演示图像
        demo_img = create_demo_image()
        
        # 2. 图像处理
        processor = ImageProcessor()
        processed_img = processor.preprocess_array(demo_img)
        
        # 3. 对象检测
        detector = FloorplanDetector()
        detections = detector.detect_objects(processed_img)
        
        # 4. 尺寸计算
        calculator = ScaleCalculator()
        calculations = calculator.calculate_sizes(detections, 100)
        
        # 5. 结果验证
        validator = DataValidator()
        validation = validator.validate_detection_results(detections)
        
        # 6. 生成报告
        report = {
            'total_objects': len(detections),
            'valid_detections': validation['valid_detections'],
            'total_area': sum(calc['real_dimensions']['area'] for calc in calculations if calc['type'] == 'room'),
            'object_counts': calculator.calculate_object_counts(detections),
            'processing_successful': True
        }
        
        print("✅ 完整处理流程演示成功")
        print(f"📊 处理报告: {json.dumps(report, ensure_ascii=False, indent=2)}")
        
        return report
        
    except Exception as e:
        print(f"❌ 完整处理流程演示失败: {str(e)}")
        return None

def main():
    """主演示函数"""
    print("=" * 60)
    print("🏗️  建筑平面图分析系统 - 功能演示")
    print("=" * 60)
    
    try:
        # 演示各个功能模块
        demo_image_processing()
        demo_object_detection()
        demo_size_calculation()
        demo_data_validation()
        demo_full_pipeline()
        
        print("\n" + "=" * 60)
        print("🎉 所有功能演示完成！")
        print("=" * 60)
        print("📝 系统功能总结:")
        print("  ✅ 图像预处理和增强")
        print("  ✅ 基于深度学习的对象检测")
        print("  ✅ 精确的尺寸计算")
        print("  ✅ 完整的数据验证")
        print("  ✅ 用户友好的Web界面")
        print("  ✅ 全面的测试覆盖")
        print("\n🚀 运行 'python start.py' 启动完整系统")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
