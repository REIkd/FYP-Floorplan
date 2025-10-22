"""
比例尺计算器
处理平面图的比例尺和尺寸计算
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)

class ScaleCalculator:
    """比例尺计算器"""
    
    def __init__(self):
        """初始化比例尺计算器"""
        # 常见比例尺
        self.common_scales = {
            50: "1:50",
            100: "1:100", 
            200: "1:200",
            500: "1:500",
            1000: "1:1000"
        }
        
        # 单位转换
        self.unit_conversions = {
            'mm': 0.001,      # 毫米到米
            'cm': 0.01,       # 厘米到米
            'm': 1.0,         # 米
            'ft': 0.3048,     # 英尺到米
            'in': 0.0254      # 英寸到米
        }
        
        logger.info("比例尺计算器初始化完成")
    
    def calculate_sizes(self, detections: List[Dict[str, Any]], scale_ratio: int = 100) -> List[Dict[str, Any]]:
        """
        计算检测对象的实际尺寸
        
        Args:
            detections: 检测结果列表
            scale_ratio: 比例尺 (如100表示1:100)
            
        Returns:
            包含尺寸计算的结果列表
        """
        try:
            size_calculations = []
            
            for detection in detections:
                calculation = self._calculate_object_size(detection, scale_ratio)
                size_calculations.append(calculation)
            
            logger.info(f"完成 {len(size_calculations)} 个对象的尺寸计算")
            return size_calculations
            
        except Exception as e:
            logger.error(f"尺寸计算失败: {str(e)}")
            return []
    
    def _calculate_object_size(self, detection: Dict[str, Any], scale_ratio: int) -> Dict[str, Any]:
        """
        计算单个对象的尺寸
        
        Args:
            detection: 检测结果
            scale_ratio: 比例尺
            
        Returns:
            尺寸计算结果
        """
        try:
            # 获取像素尺寸
            pixel_width = detection.get('width', 0)
            pixel_height = detection.get('height', 0)
            pixel_area = detection.get('area', 0)
            
            # 转换为实际尺寸 (米)
            real_width = (pixel_width / scale_ratio) * 100  # 假设图像中1单位=1cm
            real_height = (pixel_height / scale_ratio) * 100
            real_area = (pixel_area / (scale_ratio ** 2)) * 10000  # 平方米
            
            # 计算周长
            real_perimeter = 2 * (real_width + real_height)
            
            # 计算对角线长度
            real_diagonal = math.sqrt(real_width ** 2 + real_height ** 2)
            
            calculation = {
                'type': detection.get('class', 'unknown'),
                'class_id': detection.get('class_id', -1),
                'confidence': detection.get('confidence', 0),
                'pixel_dimensions': {
                    'width': pixel_width,
                    'height': pixel_height,
                    'area': pixel_area
                },
                'real_dimensions': {
                    'width': real_width,
                    'height': real_height,
                    'area': real_area,
                    'perimeter': real_perimeter,
                    'diagonal': real_diagonal
                },
                'scale_ratio': scale_ratio,
                'units': 'meters'
            }
            
            # 添加特定类型的计算
            if detection.get('class') == 'room':
                calculation.update(self._calculate_room_metrics(real_width, real_height, real_area))
            elif detection.get('class') == 'door':
                calculation.update(self._calculate_door_metrics(real_width, real_height))
            elif detection.get('class') == 'window':
                calculation.update(self._calculate_window_metrics(real_width, real_height))
            
            return calculation
            
        except Exception as e:
            logger.error(f"对象尺寸计算失败: {str(e)}")
            return {
                'type': detection.get('class', 'unknown'),
                'error': str(e),
                'real_dimensions': {'width': 0, 'height': 0, 'area': 0}
            }
    
    def _calculate_room_metrics(self, width: float, height: float, area: float) -> Dict[str, Any]:
        """计算房间相关指标"""
        return {
            'room_metrics': {
                'usable_area': area,
                'aspect_ratio': width / height if height > 0 else 0,
                'room_type': self._classify_room_type(area),
                'capacity_estimate': self._estimate_room_capacity(area)
            }
        }
    
    def _calculate_door_metrics(self, width: float, height: float) -> Dict[str, Any]:
        """计算门相关指标"""
        return {
            'door_metrics': {
                'width_category': self._classify_door_width(width),
                'height_category': self._classify_door_height(height),
                'accessibility_compliant': self._check_door_accessibility(width, height)
            }
        }
    
    def _calculate_window_metrics(self, width: float, height: float) -> Dict[str, Any]:
        """计算窗相关指标"""
        return {
            'window_metrics': {
                'size_category': self._classify_window_size(width, height),
                'lighting_area': width * height,
                'ventilation_adequate': self._check_window_ventilation(width, height)
            }
        }
    
    def _classify_room_type(self, area: float) -> str:
        """根据面积分类房间类型"""
        if area < 10:
            return "小房间"
        elif area < 20:
            return "中等房间"
        elif area < 50:
            return "大房间"
        else:
            return "超大房间"
    
    def _estimate_room_capacity(self, area: float) -> int:
        """估算房间容量 (人数)"""
        # 假设每人需要2平方米
        return max(1, int(area / 2))
    
    def _classify_door_width(self, width: float) -> str:
        """分类门宽度"""
        if width < 0.8:
            return "窄门"
        elif width < 1.0:
            return "标准门"
        else:
            return "宽门"
    
    def _classify_door_height(self, height: float) -> str:
        """分类门高度"""
        if height < 2.0:
            return "低门"
        elif height < 2.2:
            return "标准门"
        else:
            return "高门"
    
    def _check_door_accessibility(self, width: float, height: float) -> bool:
        """检查门是否符合无障碍标准"""
        return width >= 0.8 and height >= 2.0
    
    def _classify_window_size(self, width: float, height: float) -> str:
        """分类窗户尺寸"""
        area = width * height
        if area < 1.0:
            return "小窗"
        elif area < 2.0:
            return "中等窗"
        else:
            return "大窗"
    
    def _check_window_ventilation(self, width: float, height: float) -> bool:
        """检查窗户通风是否充足"""
        area = width * height
        return area >= 0.5  # 最小通风面积
    
    def calculate_total_area(self, detections: List[Dict[str, Any]], 
                           object_type: str = 'room') -> float:
        """
        计算指定类型对象的总面积
        
        Args:
            detections: 检测结果列表
            object_type: 对象类型
            
        Returns:
            总面积 (平方米)
        """
        total_area = 0.0
        
        for detection in detections:
            if detection.get('class') == object_type:
                width = detection.get('width', 0)
                height = detection.get('height', 0)
                scale_ratio = detection.get('scale_ratio', 100)
                
                real_width = (width / scale_ratio) * 100
                real_height = (height / scale_ratio) * 100
                area = real_width * real_height
                
                total_area += area
        
        return total_area
    
    def calculate_object_counts(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        统计各类对象的数量
        
        Args:
            detections: 检测结果列表
            
        Returns:
            对象数量统计字典
        """
        counts = {}
        
        for detection in detections:
            class_name = detection.get('class', 'unknown')
            counts[class_name] = counts.get(class_name, 0) + 1
        
        return counts
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        """
        转换单位
        
        Args:
            value: 数值
            from_unit: 源单位
            to_unit: 目标单位
            
        Returns:
            转换后的数值
        """
        try:
            if from_unit not in self.unit_conversions or to_unit not in self.unit_conversions:
                raise ValueError(f"不支持的单位: {from_unit} -> {to_unit}")
            
            # 先转换为米
            value_in_meters = value * self.unit_conversions[from_unit]
            
            # 再转换为目标单位
            converted_value = value_in_meters / self.unit_conversions[to_unit]
            
            return converted_value
            
        except Exception as e:
            logger.error(f"单位转换失败: {str(e)}")
            return value
    
    def get_scale_info(self, scale_ratio: int) -> Dict[str, Any]:
        """
        获取比例尺信息
        
        Args:
            scale_ratio: 比例尺数值
            
        Returns:
            比例尺信息字典
        """
        return {
            'ratio': scale_ratio,
            'display': self.common_scales.get(scale_ratio, f"1:{scale_ratio}"),
            'description': f"图上1单位 = 实际{scale_ratio}单位",
            'precision': f"±{1/scale_ratio:.3f}单位"
        }
    
    def validate_measurements(self, measurements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证测量结果的合理性
        
        Args:
            measurements: 测量结果列表
            
        Returns:
            验证后的测量结果
        """
        validated_measurements = []
        
        for measurement in measurements:
            # 基本验证
            is_valid = True
            issues = []
            
            # 检查尺寸是否为正数
            real_dims = measurement.get('real_dimensions', {})
            if real_dims.get('width', 0) <= 0:
                is_valid = False
                issues.append("宽度无效")
            
            if real_dims.get('height', 0) <= 0:
                is_valid = False
                issues.append("高度无效")
            
            # 检查尺寸是否在合理范围内
            width = real_dims.get('width', 0)
            height = real_dims.get('height', 0)
            
            if width > 100 or height > 100:  # 超过100米可能不合理
                issues.append("尺寸过大，请检查比例尺")
            
            measurement['validation'] = {
                'is_valid': is_valid,
                'issues': issues
            }
            
            validated_measurements.append(measurement)
        
        return validated_measurements
