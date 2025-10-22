"""
集成测试
测试整个系统的集成功能
"""

import unittest
import numpy as np
import tempfile
import os
import json
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.floorplan_detector import FloorplanDetector
from utils.image_processor import ImageProcessor
from utils.scale_calculator import ScaleCalculator
from utils.data_validator import DataValidator

class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = FloorplanDetector()
        self.image_processor = ImageProcessor()
        self.scale_calculator = ScaleCalculator()
        self.data_validator = DataValidator()
        
        # 创建测试图像
        self.test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_pipeline(self):
        """测试完整处理流程"""
        try:
            # 1. 图像预处理
            processed_image = self.image_processor.preprocess_array(self.test_image)
            self.assertIsNotNone(processed_image)
            
            # 2. 对象检测
            detections = self.detector.detect_objects(processed_image)
            self.assertIsInstance(detections, list)
            
            # 3. 尺寸计算
            if detections:
                size_calculations = self.scale_calculator.calculate_sizes(detections, 100)
                self.assertIsInstance(size_calculations, list)
                
                # 4. 结果验证
                validation = self.data_validator.validate_detection_results(detections)
                self.assertIsInstance(validation, dict)
                
                # 5. 尺寸计算验证
                if size_calculations:
                    scale_validation = self.data_validator.validate_scale_calculations(size_calculations)
                    self.assertIsInstance(scale_validation, dict)
            
            print("完整处理流程测试通过")
            
        except Exception as e:
            self.fail(f"完整处理流程测试失败: {str(e)}")
    
    def test_image_processing_pipeline(self):
        """测试图像处理流程"""
        try:
            # 预处理
            processed = self.image_processor.preprocess_array(self.test_image)
            
            # 增强
            enhanced = self.image_processor.enhance_image(processed, 'auto')
            
            # 去噪
            denoised = self.image_processor.remove_noise(enhanced)
            
            # 调整对比度
            adjusted = self.image_processor.adjust_contrast_brightness(denoised, 1.2, 10)
            
            self.assertIsNotNone(adjusted)
            print("图像处理流程测试通过")
            
        except Exception as e:
            self.fail(f"图像处理流程测试失败: {str(e)}")
    
    def test_detection_and_calculation_pipeline(self):
        """测试检测和计算流程"""
        try:
            # 模拟检测结果
            mock_detections = [
                {
                    'class': 'door',
                    'class_id': 0,
                    'confidence': 0.8,
                    'bbox': [100, 100, 50, 100],
                    'x': 100,
                    'y': 100,
                    'width': 50,
                    'height': 100,
                    'area': 5000
                },
                {
                    'class': 'window',
                    'class_id': 1,
                    'confidence': 0.9,
                    'bbox': [200, 150, 80, 60],
                    'x': 200,
                    'y': 150,
                    'width': 80,
                    'height': 60,
                    'area': 4800
                }
            ]
            
            # 尺寸计算
            calculations = self.scale_calculator.calculate_sizes(mock_detections, 100)
            self.assertIsInstance(calculations, list)
            self.assertEqual(len(calculations), 2)
            
            # 验证计算结果
            for calc in calculations:
                self.assertIn('real_dimensions', calc)
                self.assertIn('width', calc['real_dimensions'])
                self.assertIn('height', calc['real_dimensions'])
            
            print("检测和计算流程测试通过")
            
        except Exception as e:
            self.fail(f"检测和计算流程测试失败: {str(e)}")
    
    def test_data_validation_pipeline(self):
        """测试数据验证流程"""
        try:
            # 创建测试数据集
            test_dataset_dir = os.path.join(self.temp_dir, 'test_dataset')
            os.makedirs(test_dataset_dir, exist_ok=True)
            
            # 创建测试图像
            test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(test_dataset_dir, 'test.jpg'), test_image)
            
            # 创建测试标注文件
            with open(os.path.join(test_dataset_dir, 'test.txt'), 'w') as f:
                f.write("0 0.5 0.5 0.1 0.1\n")
                f.write("1 0.3 0.3 0.2 0.15\n")
            
            # 验证数据集
            validation_results = self.data_validator.validate_dataset(test_dataset_dir)
            self.assertIsInstance(validation_results, dict)
            self.assertIn('valid', validation_results)
            
            print("数据验证流程测试通过")
            
        except Exception as e:
            self.fail(f"数据验证流程测试失败: {str(e)}")
    
    def test_error_handling(self):
        """测试错误处理"""
        try:
            # 测试无效图像
            invalid_image = np.array([])
            
            # 应该能够处理无效输入而不崩溃
            try:
                processed = self.image_processor.preprocess_array(invalid_image)
            except Exception:
                pass  # 预期会出错
            
            # 测试空检测结果
            empty_detections = []
            calculations = self.scale_calculator.calculate_sizes(empty_detections, 100)
            self.assertEqual(len(calculations), 0)
            
            # 测试无效比例尺
            mock_detection = [{'class': 'door', 'width': 50, 'height': 100, 'area': 5000}]
            calculations = self.scale_calculator.calculate_sizes(mock_detection, 0)  # 无效比例尺
            self.assertIsInstance(calculations, list)
            
            print("错误处理测试通过")
            
        except Exception as e:
            self.fail(f"错误处理测试失败: {str(e)}")
    
    def test_performance(self):
        """测试性能"""
        import time
        
        try:
            # 测试处理时间
            start_time = time.time()
            
            # 图像预处理
            processed = self.image_processor.preprocess_array(self.test_image)
            
            # 对象检测
            detections = self.detector.detect_objects(processed)
            
            # 尺寸计算
            if detections:
                calculations = self.scale_calculator.calculate_sizes(detections, 100)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 处理时间应该在合理范围内 (比如5秒内)
            self.assertLess(processing_time, 5.0)
            
            print(f"性能测试通过，处理时间: {processing_time:.2f}秒")
            
        except Exception as e:
            self.fail(f"性能测试失败: {str(e)}")
    
    def test_memory_usage(self):
        """测试内存使用"""
        try:
            import psutil
            import os
            
            # 获取初始内存使用
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 处理多个图像
            for i in range(5):
                test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                processed = self.image_processor.preprocess_array(test_image)
                detections = self.detector.detect_objects(processed)
                
                if detections:
                    calculations = self.scale_calculator.calculate_sizes(detections, 100)
            
            # 获取最终内存使用
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # 内存增长应该在合理范围内 (比如100MB内)
            self.assertLess(memory_increase, 100)
            
            print(f"内存使用测试通过，内存增长: {memory_increase:.2f}MB")
            
        except ImportError:
            print("psutil未安装，跳过内存使用测试")
        except Exception as e:
            self.fail(f"内存使用测试失败: {str(e)}")

if __name__ == '__main__':
    unittest.main()
