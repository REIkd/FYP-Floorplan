"""
对象分类器
用于进一步分类和验证检测到的建筑对象
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ObjectClassifier:
    """建筑对象分类器"""
    
    def __init__(self):
        """初始化分类器"""
        # 对象特征规则
        self.classification_rules = {
            'door': {
                'min_aspect_ratio': 0.1,
                'max_aspect_ratio': 0.5,
                'min_area': 100,
                'max_area': 5000
            },
            'window': {
                'min_aspect_ratio': 0.2,
                'max_aspect_ratio': 0.8,
                'min_area': 200,
                'max_area': 3000
            },
            'stair': {
                'min_aspect_ratio': 0.3,
                'max_aspect_ratio': 2.0,
                'min_area': 1000,
                'max_area': 20000
            },
            'elevator': {
                'min_aspect_ratio': 0.5,
                'max_aspect_ratio': 1.5,
                'min_area': 800,
                'max_area': 4000
            },
            'room': {
                'min_aspect_ratio': 0.3,
                'max_aspect_ratio': 3.0,
                'min_area': 2000,
                'max_area': 100000
            },
            'wall': {
                'min_aspect_ratio': 0.05,
                'max_aspect_ratio': 0.3,
                'min_area': 500,
                'max_area': 15000
            },
            'column': {
                'min_aspect_ratio': 0.8,
                'max_aspect_ratio': 1.2,
                'min_area': 400,
                'max_area': 2000
            }
        }
        
        logger.info("对象分类器初始化完成")
    
    def classify_object(self, class_name: str, confidence: float, bbox: Tuple[float, float, float, float]) -> str:
        """
        分类和验证对象
        
        Args:
            class_name: 原始类别名称
            confidence: 检测置信度
            bbox: 边界框 (x, y, width, height)
            
        Returns:
            验证后的类别名称
        """
        try:
            x, y, width, height = bbox
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            # 应用分类规则
            refined_class = self._apply_classification_rules(
                class_name, area, aspect_ratio, confidence
            )
            
            return refined_class
            
        except Exception as e:
            logger.error(f"对象分类失败: {str(e)}")
            return class_name
    
    def _apply_classification_rules(self, class_name: str, area: float, aspect_ratio: float, confidence: float) -> str:
        """
        应用分类规则
        
        Args:
            class_name: 原始类别
            area: 对象面积
            aspect_ratio: 长宽比
            confidence: 置信度
            
        Returns:
            验证后的类别
        """
        if class_name not in self.classification_rules:
            return class_name
        
        rules = self.classification_rules[class_name]
        
        # 检查面积范围
        if area < rules['min_area'] or area > rules['max_area']:
            return self._find_best_match(area, aspect_ratio)
        
        # 检查长宽比
        if aspect_ratio < rules['min_aspect_ratio'] or aspect_ratio > rules['max_aspect_ratio']:
            return self._find_best_match(area, aspect_ratio)
        
        # 检查置信度
        if confidence < 0.3:
            return 'unknown'
        
        return class_name
    
    def _find_best_match(self, area: float, aspect_ratio: float) -> str:
        """
        根据特征找到最佳匹配类别
        
        Args:
            area: 对象面积
            aspect_ratio: 长宽比
            
        Returns:
            最佳匹配的类别
        """
        best_match = 'unknown'
        best_score = 0
        
        for class_name, rules in self.classification_rules.items():
            # 计算匹配分数
            area_score = self._calculate_area_score(area, rules['min_area'], rules['max_area'])
            aspect_score = self._calculate_aspect_score(aspect_ratio, rules['min_aspect_ratio'], rules['max_aspect_ratio'])
            
            total_score = (area_score + aspect_score) / 2
            
            if total_score > best_score:
                best_score = total_score
                best_match = class_name
        
        return best_match if best_score > 0.5 else 'unknown'
    
    def _calculate_area_score(self, area: float, min_area: float, max_area: float) -> float:
        """计算面积匹配分数"""
        if min_area <= area <= max_area:
            return 1.0
        elif area < min_area:
            return area / min_area
        else:
            return max_area / area
    
    def _calculate_aspect_score(self, aspect_ratio: float, min_ratio: float, max_ratio: float) -> float:
        """计算长宽比匹配分数"""
        if min_ratio <= aspect_ratio <= max_ratio:
            return 1.0
        elif aspect_ratio < min_ratio:
            return aspect_ratio / min_ratio
        else:
            return max_ratio / aspect_ratio
    
    def get_object_properties(self, class_name: str) -> Dict[str, Any]:
        """
        获取对象属性信息
        
        Args:
            class_name: 对象类别名称
            
        Returns:
            对象属性字典
        """
        properties = {
            'door': {
                'typical_width': 0.9,  # 米
                'typical_height': 2.1,
                'description': '门'
            },
            'window': {
                'typical_width': 1.2,
                'typical_height': 1.5,
                'description': '窗'
            },
            'stair': {
                'typical_width': 1.2,
                'typical_height': 2.4,
                'description': '楼梯'
            },
            'elevator': {
                'typical_width': 1.6,
                'typical_height': 2.4,
                'description': '电梯'
            },
            'room': {
                'typical_width': 3.0,
                'typical_height': 3.0,
                'description': '房间'
            },
            'wall': {
                'typical_width': 0.2,
                'typical_height': 2.8,
                'description': '墙'
            },
            'column': {
                'typical_width': 0.4,
                'typical_height': 0.4,
                'description': '柱子'
            }
        }
        
        return properties.get(class_name, {
            'typical_width': 1.0,
            'typical_height': 1.0,
            'description': '未知对象'
        })
    
    def validate_detection(self, detection: Dict[str, Any]) -> bool:
        """
        验证检测结果的有效性
        
        Args:
            detection: 检测结果字典
            
        Returns:
            是否有效
        """
        try:
            class_name = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            
            # 基本验证
            if confidence < 0.1:
                return False
            
            if len(bbox) != 4:
                return False
            
            x, y, width, height = bbox
            if width <= 0 or height <= 0:
                return False
            
            # 应用分类规则验证
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            refined_class = self._apply_classification_rules(
                class_name, area, aspect_ratio, confidence
            )
            
            return refined_class != 'unknown'
            
        except Exception as e:
            logger.error(f"检测验证失败: {str(e)}")
            return False
