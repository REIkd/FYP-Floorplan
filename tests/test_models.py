"""
模型测试
测试深度学习模型和检测器
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.floorplan_detector import FloorplanDetector
from models.yolo_detector import YOLODetector
from models.object_classifier import ObjectClassifier

class TestFloorplanDetector(unittest.TestCase):
    """测试平面图检测器"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = FloorplanDetector()
        self.test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.yolo_detector)
        self.assertIsNotNone(self.detector.classifier)
    
    def test_preprocess_image(self):
        """测试图像预处理"""
        processed = self.detector._preprocess_image(self.test_image)
        self.assertEqual(processed.shape, (640, 640, 3))
        self.assertTrue(processed.dtype == np.float32)
        self.assertTrue(np.all(processed >= 0) and np.all(processed <= 1))
    
    def test_detect_objects(self):
        """测试对象检测"""
        with patch.object(self.detector.yolo_detector, 'detect') as mock_detect:
            mock_detect.return_value = [
                {'class_id': 0, 'confidence': 0.8, 'bbox': [0.1, 0.1, 0.1, 0.1]},
                {'class_id': 1, 'confidence': 0.9, 'bbox': [0.3, 0.3, 0.1, 0.1]}
            ]
            
            results = self.detector.detect_objects(self.test_image)
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 2)
    
    def test_detect_rooms(self):
        """测试房间检测"""
        results = self.detector.detect_rooms(self.test_image)
        self.assertIsInstance(results, list)
    
    def test_get_detection_statistics(self):
        """测试检测统计"""
        detections = [
            {'class': 'door', 'confidence': 0.8, 'area': 100},
            {'class': 'window', 'confidence': 0.9, 'area': 200}
        ]
        
        stats = self.detector.get_detection_statistics(detections)
        self.assertEqual(stats['total_objects'], 2)
        self.assertEqual(stats['class_counts']['door'], 1)
        self.assertEqual(stats['class_counts']['window'], 1)

class TestYOLODetector(unittest.TestCase):
    """测试YOLO检测器"""
    
    def setUp(self):
        """设置测试环境"""
        self.detector = YOLODetector()
        self.test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.device)
    
    def test_detect(self):
        """测试检测功能"""
        results = self.detector.detect(self.test_image)
        self.assertIsInstance(results, list)
    
    def test_detect_with_confidence(self):
        """测试带置信度阈值的检测"""
        results = self.detector.detect_with_confidence(self.test_image, 0.5)
        self.assertIsInstance(results, list)
    
    def test_detect_specific_class(self):
        """测试特定类别检测"""
        results = self.detector.detect_specific_class(self.test_image, 0)
        self.assertIsInstance(results, list)
    
    def test_get_model_info(self):
        """测试获取模型信息"""
        info = self.detector.get_model_info()
        self.assertIsInstance(info, dict)
        self.assertIn('model_path', info)
        self.assertIn('device', info)

class TestObjectClassifier(unittest.TestCase):
    """测试对象分类器"""
    
    def setUp(self):
        """设置测试环境"""
        self.classifier = ObjectClassifier()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.classifier)
        self.assertIsNotNone(self.classifier.classification_rules)
    
    def test_classify_object(self):
        """测试对象分类"""
        result = self.classifier.classify_object('door', 0.8, (100, 50, 20, 40))
        self.assertIsInstance(result, str)
    
    def test_apply_classification_rules(self):
        """测试分类规则应用"""
        result = self.classifier._apply_classification_rules('door', 100, 0.5, 0.8)
        self.assertIsInstance(result, str)
    
    def test_find_best_match(self):
        """测试最佳匹配查找"""
        result = self.classifier._find_best_match(100, 0.5)
        self.assertIsInstance(result, str)
    
    def test_get_object_properties(self):
        """测试获取对象属性"""
        properties = self.classifier.get_object_properties('door')
        self.assertIsInstance(properties, dict)
        self.assertIn('typical_width', properties)
        self.assertIn('typical_height', properties)
    
    def test_validate_detection(self):
        """测试检测验证"""
        detection = {
            'class': 'door',
            'confidence': 0.8,
            'bbox': [100, 100, 50, 100]
        }
        
        is_valid = self.classifier.validate_detection(detection)
        self.assertIsInstance(is_valid, bool)

if __name__ == '__main__':
    unittest.main()
