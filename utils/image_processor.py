"""
图像处理器
处理平面图图像的预处理和后处理功能
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import logging
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

class ImageProcessor:
    """图像处理器"""
    
    def __init__(self):
        """初始化图像处理器"""
        self.target_size = (640, 640)  # YOLO输入尺寸
        self.normalization_mean = [0.485, 0.456, 0.406]
        self.normalization_std = [0.229, 0.224, 0.225]
        
        logger.info("图像处理器初始化完成")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        预处理图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            预处理后的图像数组
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 预处理
            processed_image = self._preprocess(image)
            
            logger.info(f"图像预处理完成: {image_path}")
            return processed_image
            
        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            raise
    
    def preprocess_array(self, image: np.ndarray) -> np.ndarray:
        """
        预处理图像数组
        
        Args:
            image: 输入图像数组
            
        Returns:
            预处理后的图像数组
        """
        return self._preprocess(image)
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        执行图像预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 转换为RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小
        image = cv2.resize(image, self.target_size)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def enhance_image(self, image: np.ndarray, enhancement_type: str = 'auto') -> np.ndarray:
        """
        图像增强
        
        Args:
            image: 输入图像
            enhancement_type: 增强类型 ('auto', 'contrast', 'brightness', 'sharpness')
            
        Returns:
            增强后的图像
        """
        try:
            # 转换为PIL图像
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            if enhancement_type == 'auto':
                # 自动增强
                enhanced_image = self._auto_enhance(pil_image)
            elif enhancement_type == 'contrast':
                # 对比度增强
                enhancer = ImageEnhance.Contrast(pil_image)
                enhanced_image = enhancer.enhance(1.5)
            elif enhancement_type == 'brightness':
                # 亮度增强
                enhancer = ImageEnhance.Brightness(pil_image)
                enhanced_image = enhancer.enhance(1.2)
            elif enhancement_type == 'sharpness':
                # 锐化
                enhancer = ImageEnhance.Sharpness(pil_image)
                enhanced_image = enhancer.enhance(2.0)
            else:
                enhanced_image = pil_image
            
            # 转换回numpy数组
            enhanced_array = np.array(enhanced_image)
            
            return enhanced_array
            
        except Exception as e:
            logger.error(f"图像增强失败: {str(e)}")
            return image
    
    def _auto_enhance(self, image: Image.Image) -> Image.Image:
        """自动图像增强"""
        # 对比度增强
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        # 锐化
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        return image
    
    def detect_scale_bar(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        检测图像中的比例尺
        
        Args:
            image: 输入图像
            
        Returns:
            比例尺信息字典，包含位置和长度
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 霍夫直线检测
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                   minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                # 分析直线，寻找可能的比例尺
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    # 检查是否是水平或垂直线
                    if abs(y2-y1) < 5 or abs(x2-x1) < 5:
                        if length > 50:  # 最小长度阈值
                            return {
                                'start': (x1, y1),
                                'end': (x2, y2),
                                'length': length,
                                'type': 'horizontal' if abs(y2-y1) < 5 else 'vertical'
                            }
            
            return None
            
        except Exception as e:
            logger.error(f"比例尺检测失败: {str(e)}")
            return None
    
    def extract_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        提取图像中的文本区域
        
        Args:
            image: 输入图像
            
        Returns:
            文本区域列表
        """
        try:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # 二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 5000:  # 文本区域大小范围
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # 检查长宽比，文本通常有特定的长宽比
                    if 0.1 < aspect_ratio < 10:
                        text_regions.append({
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
            
            return text_regions
            
        except Exception as e:
            logger.error(f"文本区域提取失败: {str(e)}")
            return []
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        去除图像噪声
        
        Args:
            image: 输入图像
            
        Returns:
            去噪后的图像
        """
        try:
            # 中值滤波去除椒盐噪声
            denoised = cv2.medianBlur(image, 3)
            
            # 高斯滤波去除高斯噪声
            denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
            
            return denoised
            
        except Exception as e:
            logger.error(f"去噪失败: {str(e)}")
            return image
    
    def adjust_contrast_brightness(self, image: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
        """
        调整对比度和亮度
        
        Args:
            image: 输入图像
            alpha: 对比度控制 (1.0为原始)
            beta: 亮度控制 (0为原始)
            
        Returns:
            调整后的图像
        """
        try:
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            return adjusted
            
        except Exception as e:
            logger.error(f"对比度亮度调整失败: {str(e)}")
            return image
    
    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """
        获取图像信息
        
        Args:
            image: 输入图像
            
        Returns:
            图像信息字典
        """
        try:
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            
            return {
                'width': width,
                'height': height,
                'channels': channels,
                'dtype': str(image.dtype),
                'shape': image.shape,
                'size': image.size
            }
            
        except Exception as e:
            logger.error(f"获取图像信息失败: {str(e)}")
            return {}
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    keep_aspect_ratio: bool = True) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            keep_aspect_ratio: 是否保持长宽比
            
        Returns:
            调整后的图像
        """
        try:
            if keep_aspect_ratio:
                # 保持长宽比
                h, w = image.shape[:2]
                target_w, target_h = target_size
                
                # 计算缩放比例
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                resized = cv2.resize(image, (new_w, new_h))
                
                # 创建目标尺寸的黑色背景
                result = np.zeros((target_h, target_w, image.shape[2] if len(image.shape) == 3 else 1), 
                                dtype=image.dtype)
                
                # 将调整后的图像放在中心
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
                return result
            else:
                # 直接调整到目标尺寸
                return cv2.resize(image, target_size)
                
        except Exception as e:
            logger.error(f"图像大小调整失败: {str(e)}")
            return image
