"""
数据验证器
验证数据集和检测结果的正确性
"""

import os
import json
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        """初始化数据验证器"""
        # 支持的图像格式
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        # 支持的标注格式
        self.supported_annotation_formats = {'.txt', '.json', '.xml'}
        
        # 验证规则
        self.validation_rules = {
            'image_size': {
                'min_width': 100,
                'min_height': 100,
                'max_width': 10000,
                'max_height': 10000
            },
            'annotation': {
                'min_objects': 1,
                'max_objects': 1000,
                'min_confidence': 0.1,
                'max_confidence': 1.0
            },
            'file_size': {
                'min_size': 1024,  # 1KB
                'max_size': 100 * 1024 * 1024  # 100MB
            }
        }
        
        logger.info("数据验证器初始化完成")
    
    def validate_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        验证数据集
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            验证结果字典
        """
        try:
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                return {
                    'valid': False,
                    'error': f"数据集路径不存在: {dataset_path}"
                }
            
            validation_results = {
                'valid': True,
                'total_files': 0,
                'valid_files': 0,
                'error_files': 0,
                'errors': [],
                'warnings': [],
                'statistics': {}
            }
            
            # 验证图像文件
            image_results = self._validate_images(dataset_path)
            validation_results.update(image_results)
            
            # 验证标注文件
            annotation_results = self._validate_annotations(dataset_path)
            validation_results.update(annotation_results)
            
            # 验证数据集结构
            structure_results = self._validate_dataset_structure(dataset_path)
            validation_results.update(structure_results)
            
            # 计算统计信息
            validation_results['statistics'] = self._calculate_dataset_statistics(dataset_path)
            
            logger.info(f"数据集验证完成: {validation_results['valid_files']}/{validation_results['total_files']} 文件有效")
            return validation_results
            
        except Exception as e:
            logger.error(f"数据集验证失败: {str(e)}")
            return {
                'valid': False,
                'error': str(e),
                'total_files': 0,
                'valid_files': 0,
                'error_files': 0
            }
    
    def _validate_images(self, dataset_path: Path) -> Dict[str, Any]:
        """验证图像文件"""
        image_files = []
        valid_images = 0
        error_images = 0
        errors = []
        
        # 查找所有图像文件
        for ext in self.supported_formats:
            image_files.extend(dataset_path.rglob(f"*{ext}"))
            image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
        
        for image_file in image_files:
            try:
                # 检查文件大小
                file_size = image_file.stat().st_size
                if file_size < self.validation_rules['file_size']['min_size']:
                    errors.append(f"文件过小: {image_file}")
                    error_images += 1
                    continue
                
                if file_size > self.validation_rules['file_size']['max_size']:
                    errors.append(f"文件过大: {image_file}")
                    error_images += 1
                    continue
                
                # 检查图像是否可读
                image = cv2.imread(str(image_file))
                if image is None:
                    errors.append(f"无法读取图像: {image_file}")
                    error_images += 1
                    continue
                
                # 检查图像尺寸
                height, width = image.shape[:2]
                if (width < self.validation_rules['image_size']['min_width'] or
                    height < self.validation_rules['image_size']['min_height'] or
                    width > self.validation_rules['image_size']['max_width'] or
                    height > self.validation_rules['image_size']['max_height']):
                    errors.append(f"图像尺寸不符合要求: {image_file} ({width}x{height})")
                    error_images += 1
                    continue
                
                valid_images += 1
                
            except Exception as e:
                errors.append(f"验证图像时出错: {image_file} - {str(e)}")
                error_images += 1
        
        return {
            'total_files': len(image_files),
            'valid_files': valid_images,
            'error_files': error_images,
            'errors': errors
        }
    
    def _validate_annotations(self, dataset_path: Path) -> Dict[str, Any]:
        """验证标注文件"""
        annotation_files = []
        valid_annotations = 0
        error_annotations = 0
        errors = []
        
        # 查找标注文件
        for ext in self.supported_annotation_formats:
            annotation_files.extend(dataset_path.rglob(f"*{ext}"))
        
        for annotation_file in annotation_files:
            try:
                if annotation_file.suffix == '.txt':
                    # YOLO格式标注
                    if not self._validate_yolo_annotation(annotation_file):
                        errors.append(f"YOLO标注格式错误: {annotation_file}")
                        error_annotations += 1
                        continue
                elif annotation_file.suffix == '.json':
                    # JSON格式标注
                    if not self._validate_json_annotation(annotation_file):
                        errors.append(f"JSON标注格式错误: {annotation_file}")
                        error_annotations += 1
                        continue
                
                valid_annotations += 1
                
            except Exception as e:
                errors.append(f"验证标注时出错: {annotation_file} - {str(e)}")
                error_annotations += 1
        
        return {
            'annotation_files': len(annotation_files),
            'valid_annotations': valid_annotations,
            'error_annotations': error_annotations,
            'annotation_errors': errors
        }
    
    def _validate_yolo_annotation(self, annotation_file: Path) -> bool:
        """验证YOLO格式标注"""
        try:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    return False
                
                class_id, x, y, w, h = parts
                
                # 验证类别ID
                try:
                    class_id = int(class_id)
                    if class_id < 0:
                        return False
                except ValueError:
                    return False
                
                # 验证坐标 (应该在0-1之间)
                try:
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        return False
                except ValueError:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_json_annotation(self, annotation_file: Path) -> bool:
        """验证JSON格式标注"""
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查基本结构
            if not isinstance(data, dict):
                return False
            
            # 检查必要字段
            required_fields = ['images', 'annotations']
            for field in required_fields:
                if field not in data:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_dataset_structure(self, dataset_path: Path) -> Dict[str, Any]:
        """验证数据集结构"""
        structure_issues = []
        
        # 检查是否有训练/验证/测试分割
        train_dir = dataset_path / 'train'
        val_dir = dataset_path / 'val'
        test_dir = dataset_path / 'test'
        
        if not train_dir.exists():
            structure_issues.append("缺少训练集目录")
        
        if not val_dir.exists():
            structure_issues.append("缺少验证集目录")
        
        # 检查是否有类别文件
        classes_file = dataset_path / 'classes.txt'
        if not classes_file.exists():
            structure_issues.append("缺少类别文件 (classes.txt)")
        
        return {
            'structure_issues': structure_issues,
            'has_train_split': train_dir.exists(),
            'has_val_split': val_dir.exists(),
            'has_test_split': test_dir.exists(),
            'has_classes_file': classes_file.exists()
        }
    
    def _calculate_dataset_statistics(self, dataset_path: Path) -> Dict[str, Any]:
        """计算数据集统计信息"""
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'class_distribution': {},
            'image_size_distribution': {},
            'average_objects_per_image': 0
        }
        
        try:
            # 统计图像数量
            image_count = 0
            for ext in self.supported_formats:
                image_count += len(list(dataset_path.rglob(f"*{ext}")))
                image_count += len(list(dataset_path.rglob(f"*{ext.upper()}")))
            
            stats['total_images'] = image_count
            
            # 统计标注数量
            annotation_count = 0
            for ext in self.supported_annotation_formats:
                annotation_count += len(list(dataset_path.rglob(f"*{ext}")))
            
            stats['total_annotations'] = annotation_count
            
            if image_count > 0:
                stats['average_objects_per_image'] = annotation_count / image_count
            
        except Exception as e:
            logger.error(f"计算数据集统计信息失败: {str(e)}")
        
        return stats
    
    def validate_detection_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证检测结果
        
        Args:
            results: 检测结果列表
            
        Returns:
            验证结果
        """
        validation_results = {
            'valid': True,
            'total_detections': len(results),
            'valid_detections': 0,
            'invalid_detections': 0,
            'issues': []
        }
        
        for i, result in enumerate(results):
            is_valid = True
            issues = []
            
            # 检查必要字段
            required_fields = ['class', 'confidence', 'bbox']
            for field in required_fields:
                if field not in result:
                    is_valid = False
                    issues.append(f"缺少必要字段: {field}")
            
            # 检查置信度
            confidence = result.get('confidence', 0)
            if not (0 <= confidence <= 1):
                is_valid = False
                issues.append(f"置信度超出范围: {confidence}")
            
            # 检查边界框
            bbox = result.get('bbox', [])
            if len(bbox) != 4:
                is_valid = False
                issues.append(f"边界框格式错误: {bbox}")
            else:
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    is_valid = False
                    issues.append(f"边界框尺寸无效: {bbox}")
            
            if is_valid:
                validation_results['valid_detections'] += 1
            else:
                validation_results['invalid_detections'] += 1
                validation_results['issues'].extend([f"检测 {i}: {issue}" for issue in issues])
        
        validation_results['valid'] = validation_results['invalid_detections'] == 0
        
        return validation_results
    
    def validate_scale_calculations(self, calculations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证尺寸计算结果
        
        Args:
            calculations: 尺寸计算结果列表
            
        Returns:
            验证结果
        """
        validation_results = {
            'valid': True,
            'total_calculations': len(calculations),
            'valid_calculations': 0,
            'invalid_calculations': 0,
            'issues': []
        }
        
        for i, calc in enumerate(calculations):
            is_valid = True
            issues = []
            
            # 检查尺寸数据
            real_dims = calc.get('real_dimensions', {})
            width = real_dims.get('width', 0)
            height = real_dims.get('height', 0)
            area = real_dims.get('area', 0)
            
            if width <= 0:
                is_valid = False
                issues.append(f"宽度无效: {width}")
            
            if height <= 0:
                is_valid = False
                issues.append(f"高度无效: {height}")
            
            if area <= 0:
                is_valid = False
                issues.append(f"面积无效: {area}")
            
            # 检查尺寸合理性
            if width > 100 or height > 100:
                issues.append(f"尺寸过大，请检查比例尺: {width}x{height}")
            
            if is_valid:
                validation_results['valid_calculations'] += 1
            else:
                validation_results['invalid_calculations'] += 1
                validation_results['issues'].extend([f"计算 {i}: {issue}" for issue in issues])
        
        validation_results['valid'] = validation_results['invalid_calculations'] == 0
        
        return validation_results
