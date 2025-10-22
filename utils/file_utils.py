"""
文件工具
处理文件操作和路径管理
"""

import os
import shutil
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class FileUtils:
    """文件工具类"""
    
    def __init__(self):
        """初始化文件工具"""
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        self.supported_annotation_formats = {'.txt', '.json', '.xml'}
        
        logger.info("文件工具初始化完成")
    
    def create_directory_structure(self, base_path: str) -> bool:
        """
        创建数据集目录结构
        
        Args:
            base_path: 基础路径
            
        Returns:
            是否创建成功
        """
        try:
            base_path = Path(base_path)
            
            # 创建主要目录
            directories = [
                'images/train',
                'images/val', 
                'images/test',
                'annotations/train',
                'annotations/val',
                'annotations/test',
                'models',
                'results',
                'logs'
            ]
            
            for directory in directories:
                dir_path = base_path / directory
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # 创建配置文件
            self._create_config_files(base_path)
            
            logger.info(f"目录结构创建完成: {base_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建目录结构失败: {str(e)}")
            return False
    
    def _create_config_files(self, base_path: Path):
        """创建配置文件"""
        # 创建类别文件
        classes_file = base_path / 'classes.txt'
        if not classes_file.exists():
            with open(classes_file, 'w', encoding='utf-8') as f:
                f.write("door\nwindow\nstair\nelevator\nroom\nwall\ncolumn\nbathroom\nkitchen\nbalcony\ncorridor")
        
        # 创建数据集配置文件
        config = {
            'dataset_name': 'floorplan_dataset',
            'version': '1.0',
            'description': '建筑平面图数据集',
            'classes': [
                'door', 'window', 'stair', 'elevator', 'room',
                'wall', 'column', 'bathroom', 'kitchen', 'balcony', 'corridor'
            ],
            'image_formats': list(self.supported_image_formats),
            'annotation_format': 'yolo'
        }
        
        config_file = base_path / 'dataset_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def organize_dataset(self, source_path: str, target_path: str, 
                        train_ratio: float = 0.7, val_ratio: float = 0.2) -> bool:
        """
        整理数据集
        
        Args:
            source_path: 源路径
            target_path: 目标路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            是否整理成功
        """
        try:
            source_path = Path(source_path)
            target_path = Path(target_path)
            
            # 创建目标目录结构
            self.create_directory_structure(target_path)
            
            # 获取所有图像文件
            image_files = []
            for ext in self.supported_image_formats:
                image_files.extend(source_path.rglob(f"*{ext}"))
                image_files.extend(source_path.rglob(f"*{ext.upper()}"))
            
            # 随机打乱
            import random
            random.shuffle(image_files)
            
            # 计算分割点
            total_files = len(image_files)
            train_count = int(total_files * train_ratio)
            val_count = int(total_files * val_ratio)
            
            # 复制文件到对应目录
            for i, image_file in enumerate(image_files):
                if i < train_count:
                    split = 'train'
                elif i < train_count + val_count:
                    split = 'val'
                else:
                    split = 'test'
                
                # 复制图像文件
                target_image_dir = target_path / 'images' / split
                shutil.copy2(image_file, target_image_dir)
                
                # 复制对应的标注文件
                annotation_file = self._find_annotation_file(image_file)
                if annotation_file and annotation_file.exists():
                    target_annotation_dir = target_path / 'annotations' / split
                    shutil.copy2(annotation_file, target_annotation_dir)
            
            logger.info(f"数据集整理完成: {total_files} 个文件")
            return True
            
        except Exception as e:
            logger.error(f"数据集整理失败: {str(e)}")
            return False
    
    def _find_annotation_file(self, image_file: Path) -> Optional[Path]:
        """查找对应的标注文件"""
        base_name = image_file.stem
        
        for ext in self.supported_annotation_formats:
            annotation_file = image_file.parent / f"{base_name}{ext}"
            if annotation_file.exists():
                return annotation_file
        
        return None
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        获取文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {'error': '文件不存在'}
            
            stat = file_path.stat()
            
            info = {
                'name': file_path.name,
                'path': str(file_path),
                'size': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'extension': file_path.suffix.lower(),
                'is_image': file_path.suffix.lower() in self.supported_image_formats,
                'is_annotation': file_path.suffix.lower() in self.supported_annotation_formats
            }
            
            # 计算文件哈希
            info['hash'] = self._calculate_file_hash(file_path)
            
            return info
            
        except Exception as e:
            logger.error(f"获取文件信息失败: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def find_duplicate_files(self, directory: str) -> List[List[str]]:
        """
        查找重复文件
        
        Args:
            directory: 目录路径
            
        Returns:
            重复文件组列表
        """
        try:
            directory = Path(directory)
            file_hashes = {}
            duplicates = []
            
            # 计算所有文件的哈希值
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    file_hash = self._calculate_file_hash(file_path)
                    if file_hash:
                        if file_hash in file_hashes:
                            file_hashes[file_hash].append(str(file_path))
                        else:
                            file_hashes[file_hash] = [str(file_path)]
            
            # 找出重复文件
            for file_hash, files in file_hashes.items():
                if len(files) > 1:
                    duplicates.append(files)
            
            logger.info(f"找到 {len(duplicates)} 组重复文件")
            return duplicates
            
        except Exception as e:
            logger.error(f"查找重复文件失败: {str(e)}")
            return []
    
    def clean_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        清理数据集
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            清理结果
        """
        try:
            dataset_path = Path(dataset_path)
            results = {
                'removed_files': 0,
                'removed_directories': 0,
                'freed_space': 0,
                'errors': []
            }
            
            # 查找并删除重复文件
            duplicates = self.find_duplicate_files(dataset_path)
            for duplicate_group in duplicates:
                # 保留第一个文件，删除其余的
                for file_path in duplicate_group[1:]:
                    try:
                        file_size = Path(file_path).stat().st_size
                        Path(file_path).unlink()
                        results['removed_files'] += 1
                        results['freed_space'] += file_size
                    except Exception as e:
                        results['errors'].append(f"删除文件失败: {file_path} - {str(e)}")
            
            # 删除空目录
            for directory in dataset_path.rglob('*'):
                if directory.is_dir() and not any(directory.iterdir()):
                    try:
                        directory.rmdir()
                        results['removed_directories'] += 1
                    except Exception as e:
                        results['errors'].append(f"删除目录失败: {directory} - {str(e)}")
            
            results['freed_space_mb'] = results['freed_space'] / (1024 * 1024)
            
            logger.info(f"数据集清理完成: 删除 {results['removed_files']} 个文件")
            return results
            
        except Exception as e:
            logger.error(f"数据集清理失败: {str(e)}")
            return {'error': str(e)}
    
    def backup_dataset(self, source_path: str, backup_path: str) -> bool:
        """
        备份数据集
        
        Args:
            source_path: 源路径
            backup_path: 备份路径
            
        Returns:
            是否备份成功
        """
        try:
            source_path = Path(source_path)
            backup_path = Path(backup_path)
            
            # 创建备份目录
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # 复制整个目录
            shutil.copytree(source_path, backup_path / source_path.name, dirs_exist_ok=True)
            
            logger.info(f"数据集备份完成: {source_path} -> {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"数据集备份失败: {str(e)}")
            return False
    
    def export_results(self, results: List[Dict[str, Any]], output_path: str, 
                      format: str = 'json') -> bool:
        """
        导出结果
        
        Args:
            results: 结果列表
            output_path: 输出路径
            format: 导出格式 ('json', 'csv', 'txt')
            
        Returns:
            是否导出成功
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            elif format == 'csv':
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False, encoding='utf-8')
            
            elif format == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for i, result in enumerate(results):
                        f.write(f"结果 {i+1}:\n")
                        f.write(f"  类别: {result.get('class', 'unknown')}\n")
                        f.write(f"  置信度: {result.get('confidence', 0):.3f}\n")
                        f.write(f"  位置: {result.get('bbox', [])}\n")
                        f.write("\n")
            
            logger.info(f"结果导出完成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"结果导出失败: {str(e)}")
            return False
    
    def get_dataset_statistics(self, dataset_path: str) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Args:
            dataset_path: 数据集路径
            
        Returns:
            统计信息字典
        """
        try:
            dataset_path = Path(dataset_path)
            stats = {
                'total_images': 0,
                'total_annotations': 0,
                'image_formats': {},
                'annotation_formats': {},
                'directory_structure': {},
                'total_size_mb': 0
            }
            
            # 统计图像文件
            for ext in self.supported_image_formats:
                count = len(list(dataset_path.rglob(f"*{ext}"))) + len(list(dataset_path.rglob(f"*{ext.upper()}")))
                if count > 0:
                    stats['image_formats'][ext] = count
                    stats['total_images'] += count
            
            # 统计标注文件
            for ext in self.supported_annotation_formats:
                count = len(list(dataset_path.rglob(f"*{ext}")))
                if count > 0:
                    stats['annotation_formats'][ext] = count
                    stats['total_annotations'] += count
            
            # 统计目录结构
            for split in ['train', 'val', 'test']:
                split_dir = dataset_path / split
                if split_dir.exists():
                    image_count = len(list(split_dir.rglob('*'))) if split_dir.is_dir() else 0
                    stats['directory_structure'][split] = image_count
            
            # 计算总大小
            total_size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
            stats['total_size_mb'] = total_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"获取数据集统计信息失败: {str(e)}")
            return {'error': str(e)}
