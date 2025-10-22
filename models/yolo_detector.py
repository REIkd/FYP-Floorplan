"""
YOLO对象检测器
基于YOLOv8的平面图对象检测
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics包未安装，将使用模拟检测")

logger = logging.getLogger(__name__)

class YOLODetector:
    """YOLO对象检测器"""
    
    def __init__(self, model_path: str = "models/yolo_floorplan.pt"):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 尝试加载模型
        self._load_model()
        
        logger.info(f"YOLO检测器初始化完成，设备: {self.device}")
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            if YOLO_AVAILABLE and Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                logger.info("YOLO模型加载成功")
            else:
                logger.warning("YOLO模型文件不存在，将使用模拟检测")
                self.model = None
        except Exception as e:
            logger.error(f"YOLO模型加载失败: {str(e)}")
            self.model = None
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        执行对象检测
        
        Args:
            image: 输入图像 (numpy array)
            
        Returns:
            检测结果列表
        """
        try:
            if self.model is not None:
                return self._real_detection(image)
            else:
                return self._mock_detection(image)
        except Exception as e:
            logger.error(f"检测失败: {str(e)}")
            return []
    
    def _real_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """真实YOLO检测"""
        try:
            # 执行YOLO检测
            results = self.model(image, conf=0.5, iou=0.45)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 提取检测信息
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 转换为相对坐标
                        img_height, img_width = image.shape[:2]
                        x = x1 / img_width
                        y = y1 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        detection = {
                            'class_id': class_id,
                            'confidence': float(confidence),
                            'bbox': [x, y, width, height],
                            'xyxy': [x1, y1, x2, y2]
                        }
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO检测失败: {str(e)}")
            return []
    
    def _mock_detection(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """模拟检测结果（用于演示）"""
        # 生成一些模拟的检测结果
        mock_detections = [
            {
                'class_id': 0,  # door
                'confidence': 0.85,
                'bbox': [0.1, 0.2, 0.05, 0.1],
                'xyxy': [64, 128, 96, 192]
            },
            {
                'class_id': 1,  # window
                'confidence': 0.78,
                'bbox': [0.3, 0.15, 0.08, 0.06],
                'xyxy': [192, 96, 243, 134]
            },
            {
                'class_id': 4,  # room
                'confidence': 0.92,
                'bbox': [0.2, 0.3, 0.4, 0.5],
                'xyxy': [128, 192, 384, 512]
            },
            {
                'class_id': 2,  # stair
                'confidence': 0.73,
                'bbox': [0.6, 0.4, 0.15, 0.2],
                'xyxy': [384, 256, 480, 384]
            }
        ]
        
        logger.info("使用模拟检测结果")
        return mock_detections
    
    def detect_with_confidence(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        带置信度阈值的检测
        
        Args:
            image: 输入图像
            conf_threshold: 置信度阈值
            
        Returns:
            过滤后的检测结果
        """
        detections = self.detect(image)
        return [det for det in detections if det['confidence'] >= conf_threshold]
    
    def detect_specific_class(self, image: np.ndarray, class_id: int) -> List[Dict[str, Any]]:
        """
        检测特定类别的对象
        
        Args:
            image: 输入图像
            class_id: 目标类别ID
            
        Returns:
            特定类别的检测结果
        """
        detections = self.detect(image)
        return [det for det in detections if det['class_id'] == class_id]
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model is not None:
            return {
                'model_path': self.model_path,
                'device': self.device,
                'model_loaded': True,
                'classes': getattr(self.model, 'names', {})
            }
        else:
            return {
                'model_path': self.model_path,
                'device': self.device,
                'model_loaded': False,
                'classes': {}
            }
