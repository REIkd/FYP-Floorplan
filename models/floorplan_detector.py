"""
建筑平面图检测器
基于YOLO的平面图对象检测系统
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path

from .yolo_detector import YOLODetector
from .object_classifier import ObjectClassifier

logger = logging.getLogger(__name__)

class FloorplanDetector:
    """平面图对象检测器"""
    
    def __init__(self, model_path: str = "models/yolo_floorplan.pt"):
        """
        初始化检测器
        
        Args:
            model_path: YOLO模型文件路径
        """
        self.model_path = model_path
        self.yolo_detector = YOLODetector(model_path)
        self.classifier = ObjectClassifier()
        
        # 建筑对象类别定义
        self.class_names = {
            0: 'door',           # 门
            1: 'window',         # 窗
            2: 'stair',          # 楼梯
            3: 'elevator',       # 电梯
            4: 'room',           # 房间
            5: 'wall',           # 墙
            6: 'column',         # 柱子
            7: 'bathroom',       # 卫生间
            8: 'kitchen',        # 厨房
            9: 'balcony',        # 阳台
            10: 'corridor'       # 走廊
        }
        
        logger.info("平面图检测器初始化完成")
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测平面图中的对象
        
        Args:
            image: 输入图像 (numpy array)
            
        Returns:
            检测结果列表，每个结果包含类别、置信度、边界框等信息
        """
        try:
            # 预处理图像
            processed_image = self._preprocess_image(image)
            
            # YOLO检测
            yolo_results = self.yolo_detector.detect(processed_image)
            
            # 后处理和分类
            detection_results = self._postprocess_detections(yolo_results, image.shape)
            
            logger.info(f"检测到 {len(detection_results)} 个对象")
            return detection_results
            
        except Exception as e:
            logger.error(f"对象检测失败: {str(e)}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像
        
        Args:
            image: 原始图像
            
        Returns:
            预处理后的图像
        """
        # 转换为RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小 (YOLO输入尺寸)
        target_size = (640, 640)
        image = cv2.resize(image, target_size)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def _postprocess_detections(self, yolo_results: List, image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
        """
        后处理检测结果
        
        Args:
            yolo_results: YOLO原始检测结果
            image_shape: 原始图像尺寸 (height, width, channels)
            
        Returns:
            处理后的检测结果
        """
        detection_results = []
        
        for result in yolo_results:
            # 提取检测信息
            class_id = int(result['class_id'])
            confidence = float(result['confidence'])
            bbox = result['bbox']  # [x, y, width, height]
            
            # 转换坐标到原始图像尺寸
            orig_height, orig_width = image_shape[:2]
            x = bbox[0] * orig_width
            y = bbox[1] * orig_height
            width = bbox[2] * orig_width
            height = bbox[3] * orig_height
            
            # 获取类别名称
            class_name = self.class_names.get(class_id, 'unknown')
            
            # 进一步分类和验证
            refined_class = self.classifier.classify_object(
                class_name, confidence, (x, y, width, height)
            )
            
            detection_result = {
                'class': refined_class,
                'class_id': class_id,
                'confidence': confidence,
                'bbox': [x, y, width, height],
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'area': width * height
            }
            
            detection_results.append(detection_result)
        
        # 按置信度排序
        detection_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detection_results
    
    def detect_rooms(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        专门检测房间区域
        
        Args:
            image: 输入图像
            
        Returns:
            房间检测结果
        """
        try:
            # 使用轮廓检测方法识别房间
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 二值化
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # 形态学操作
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rooms = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 过滤小区域
                    x, y, w, h = cv2.boundingRect(contour)
                    rooms.append({
                        'class': 'room',
                        'class_id': 4,
                        'confidence': 0.8,
                        'bbox': [x, y, w, h],
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'area': area
                    })
            
            return rooms
            
        except Exception as e:
            logger.error(f"房间检测失败: {str(e)}")
            return []
    
    def get_detection_statistics(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取检测统计信息
        
        Args:
            detections: 检测结果列表
            
        Returns:
            统计信息字典
        """
        if not detections:
            return {
                'total_objects': 0,
                'class_counts': {},
                'average_confidence': 0,
                'total_area': 0
            }
        
        class_counts = {}
        total_confidence = 0
        total_area = 0
        
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += detection['confidence']
            total_area += detection['area']
        
        return {
            'total_objects': len(detections),
            'class_counts': class_counts,
            'average_confidence': total_confidence / len(detections),
            'total_area': total_area
        }
