"""
工具模块测试
测试图像处理、尺寸计算、数据验证等功能
"""

import unittest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processor import ImageProcessor
from utils.scale_calculator import ScaleCalculator
from utils.data_validator import DataValidator
from utils.file_utils import FileUtils

class TestImageProcessor(unittest.TestCase):
    """测试图像处理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.processor = ImageProcessor()
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.target_size, (640, 640))
    
    def test_preprocess_array(self):
        """测试数组预处理"""
        processed = self.processor.preprocess_array(self.test_image)
        self.assertEqual(processed.shape, (640, 640, 3))
        self.assertTrue(processed.dtype == np.float32)
    
    def test_enhance_image(self):
        """测试图像增强"""
        enhanced = self.processor.enhance_image(self.test_image, 'contrast')
        self.assertEqual(enhanced.shape, self.test_image.shape)
    
    def test_detect_scale_bar(self):
        """测试比例尺检测"""
        # 创建一个简单的测试图像
        test_img = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.line(test_img, (10, 50), (50, 50), (255, 255, 255), 2)
        
        scale_info = self.processor.detect_scale_bar(test_img)
        # 由于是模拟测试，可能返回None
        self.assertTrue(scale_info is None or isinstance(scale_info, dict))
    
    def test_extract_text_regions(self):
        """测试文本区域提取"""
        regions = self.processor.extract_text_regions(self.test_image)
        self.assertIsInstance(regions, list)
    
    def test_remove_noise(self):
        """测试去噪"""
        denoised = self.processor.remove_noise(self.test_image)
        self.assertEqual(denoised.shape, self.test_image.shape)
    
    def test_adjust_contrast_brightness(self):
        """测试对比度亮度调整"""
        adjusted = self.processor.adjust_contrast_brightness(self.test_image, 1.2, 10)
        self.assertEqual(adjusted.shape, self.test_image.shape)
    
    def test_get_image_info(self):
        """测试获取图像信息"""
        info = self.processor.get_image_info(self.test_image)
        self.assertIsInstance(info, dict)
        self.assertIn('width', info)
        self.assertIn('height', info)
    
    def test_resize_image(self):
        """测试图像大小调整"""
        resized = self.processor.resize_image(self.test_image, (320, 240))
        self.assertEqual(resized.shape[:2], (240, 320))

class TestScaleCalculator(unittest.TestCase):
    """测试比例尺计算器"""
    
    def setUp(self):
        """设置测试环境"""
        self.calculator = ScaleCalculator()
        self.test_detections = [
            {
                'class': 'door',
                'width': 50,
                'height': 100,
                'area': 5000,
                'confidence': 0.8
            },
            {
                'class': 'room',
                'width': 200,
                'height': 300,
                'area': 60000,
                'confidence': 0.9
            }
        ]
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.calculator)
        self.assertIsNotNone(self.calculator.common_scales)
    
    def test_calculate_sizes(self):
        """测试尺寸计算"""
        results = self.calculator.calculate_sizes(self.test_detections, 100)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertIn('real_dimensions', result)
            self.assertIn('width', result['real_dimensions'])
            self.assertIn('height', result['real_dimensions'])
    
    def test_calculate_object_size(self):
        """测试单个对象尺寸计算"""
        detection = self.test_detections[0]
        result = self.calculator._calculate_object_size(detection, 100)
        
        self.assertIn('real_dimensions', result)
        self.assertIn('width', result['real_dimensions'])
        self.assertIn('height', result['real_dimensions'])
    
    def test_calculate_total_area(self):
        """测试总面积计算"""
        total_area = self.calculator.calculate_total_area(self.test_detections, 'room')
        self.assertIsInstance(total_area, float)
        self.assertGreater(total_area, 0)
    
    def test_calculate_object_counts(self):
        """测试对象数量统计"""
        counts = self.calculator.calculate_object_counts(self.test_detections)
        self.assertIsInstance(counts, dict)
        self.assertEqual(counts['door'], 1)
        self.assertEqual(counts['room'], 1)
    
    def test_convert_units(self):
        """测试单位转换"""
        # 测试厘米到米的转换
        result = self.calculator.convert_units(100, 'cm', 'm')
        self.assertAlmostEqual(result, 1.0, places=2)
        
        # 测试英尺到米的转换
        result = self.calculator.convert_units(1, 'ft', 'm')
        self.assertAlmostEqual(result, 0.3048, places=4)
    
    def test_get_scale_info(self):
        """测试获取比例尺信息"""
        info = self.calculator.get_scale_info(100)
        self.assertIsInstance(info, dict)
        self.assertIn('ratio', info)
        self.assertIn('display', info)
    
    def test_validate_measurements(self):
        """测试测量结果验证"""
        measurements = [
            {
                'real_dimensions': {'width': 1.0, 'height': 2.0, 'area': 2.0}
            },
            {
                'real_dimensions': {'width': -1.0, 'height': 2.0, 'area': -2.0}
            }
        ]
        
        validated = self.calculator.validate_measurements(measurements)
        self.assertEqual(len(validated), 2)
        
        # 第一个应该有效，第二个应该无效
        self.assertTrue(validated[0]['validation']['is_valid'])
        self.assertFalse(validated[1]['validation']['is_valid'])

class TestDataValidator(unittest.TestCase):
    """测试数据验证器"""
    
    def setUp(self):
        """设置测试环境"""
        self.validator = DataValidator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.validator)
        self.assertIsNotNone(self.validator.supported_formats)
    
    def test_validate_dataset(self):
        """测试数据集验证"""
        # 创建测试数据集结构
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'annotations'), exist_ok=True)
        
        # 创建测试图像文件
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(self.temp_dir, 'images', 'test.jpg'), test_image)
        
        # 创建测试标注文件
        with open(os.path.join(self.temp_dir, 'annotations', 'test.txt'), 'w') as f:
            f.write("0 0.5 0.5 0.1 0.1\n")
        
        results = self.validator.validate_dataset(self.temp_dir)
        self.assertIsInstance(results, dict)
        self.assertIn('valid', results)
    
    def test_validate_detection_results(self):
        """测试检测结果验证"""
        valid_results = [
            {
                'class': 'door',
                'confidence': 0.8,
                'bbox': [100, 100, 50, 100]
            }
        ]
        
        invalid_results = [
            {
                'class': 'door',
                'confidence': 1.5,  # 无效置信度
                'bbox': [100, 100, 50, 100]
            }
        ]
        
        valid_validation = self.validator.validate_detection_results(valid_results)
        invalid_validation = self.validator.validate_detection_results(invalid_results)
        
        self.assertTrue(valid_validation['valid'])
        self.assertFalse(invalid_validation['valid'])
    
    def test_validate_scale_calculations(self):
        """测试尺寸计算验证"""
        valid_calculations = [
            {
                'real_dimensions': {'width': 1.0, 'height': 2.0, 'area': 2.0}
            }
        ]
        
        invalid_calculations = [
            {
                'real_dimensions': {'width': -1.0, 'height': 2.0, 'area': -2.0}
            }
        ]
        
        valid_validation = self.validator.validate_scale_calculations(valid_calculations)
        invalid_validation = self.validator.validate_scale_calculations(invalid_calculations)
        
        self.assertTrue(valid_validation['valid'])
        self.assertFalse(invalid_validation['valid'])

class TestFileUtils(unittest.TestCase):
    """测试文件工具"""
    
    def setUp(self):
        """设置测试环境"""
        self.file_utils = FileUtils()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.file_utils)
        self.assertIsNotNone(self.file_utils.supported_image_formats)
    
    def test_create_directory_structure(self):
        """测试创建目录结构"""
        result = self.file_utils.create_directory_structure(self.temp_dir)
        self.assertTrue(result)
        
        # 检查关键目录是否存在
        expected_dirs = ['images/train', 'images/val', 'annotations/train']
        for dir_path in expected_dirs:
            full_path = os.path.join(self.temp_dir, dir_path)
            self.assertTrue(os.path.exists(full_path))
    
    def test_get_file_info(self):
        """测试获取文件信息"""
        # 创建测试文件
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write("test content")
        
        info = self.file_utils.get_file_info(test_file)
        self.assertIsInstance(info, dict)
        self.assertIn('name', info)
        self.assertIn('size', info)
    
    def test_get_dataset_statistics(self):
        """测试获取数据集统计信息"""
        # 创建测试数据集
        os.makedirs(os.path.join(self.temp_dir, 'images'), exist_ok=True)
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(self.temp_dir, 'images', 'test.jpg'), test_image)
        
        stats = self.file_utils.get_dataset_statistics(self.temp_dir)
        self.assertIsInstance(stats, dict)
        self.assertIn('total_images', stats)

if __name__ == '__main__':
    unittest.main()
