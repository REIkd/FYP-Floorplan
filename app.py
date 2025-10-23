"""
建筑平面图分析系统 - 主应用入口
基于深度学习的平面图识别和分析系统
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
from datetime import datetime
import logging
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 尝试导入可选依赖
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("警告: flask_cors未安装，CORS功能将不可用")

# 导入自定义模块
try:
    from models.floorplan_detector import FloorplanDetector
    from utils.image_processor import ImageProcessor
    from utils.scale_calculator import ScaleCalculator
    from utils.data_validator import DataValidator
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"警告: 某些模块导入失败: {e}")
    MODULES_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
if CORS_AVAILABLE:
    CORS(app)

# 配置
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ANALYSIS_FOLDER'] = 'analysis_images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 确保必要的文件夹存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER'], app.config['ANALYSIS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# 初始化组件
if MODULES_AVAILABLE:
    detector = FloorplanDetector()
    image_processor = ImageProcessor()
    scale_calculator = ScaleCalculator()
    data_validator = DataValidator()
else:
    detector = None
    image_processor = None
    scale_calculator = None
    data_validator = None

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if file and allowed_file(file.filename):
            # 保存上传的文件
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"文件上传成功: {filename}")
            return jsonify({
                'success': True,
                'filename': filename,
                'message': '文件上传成功'
            })
        else:
            return jsonify({'error': '不支持的文件格式'}), 400
            
    except Exception as e:
        logger.error(f"文件上传错误: {str(e)}")
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_floorplan():
    """分析平面图"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        scale_ratio = data.get('scale_ratio', 100)  # 默认1:100
        
        if not filename:
            return jsonify({'error': '缺少文件名'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': '文件不存在'}), 404
        
        # 处理图像
        if image_processor:
            processed_image = image_processor.preprocess_image(filepath)
        else:
            # 模拟处理
            processed_image = None
        
        # 检测对象
        if detector:
            detection_results = detector.detect_objects(processed_image)
        else:
            # 模拟检测结果
            detection_results = [
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
        
        # 计算尺寸
        if scale_calculator:
            size_calculations = scale_calculator.calculate_sizes(
                detection_results, scale_ratio
            )
        else:
            # 模拟尺寸计算
            size_calculations = []
            for detection in detection_results:
                width = detection.get('width', 0)
                height = detection.get('height', 0)
                real_width = (width / scale_ratio) * 100
                real_height = (height / scale_ratio) * 100
                real_area = real_width * real_height
                
                size_calculations.append({
                    'type': detection.get('class', 'unknown'),
                    'real_dimensions': {
                        'width': real_width,
                        'height': real_height,
                        'area': real_area
                    }
                })
        
        # 生成统计报告
        statistics = generate_statistics(detection_results, size_calculations)
        
        # 绘制检测框并保存分析图片
        analysis_image_path = draw_detection_boxes(filepath, detection_results, filename)
        
        # 保存结果
        result_filename = f"result_{filename.replace('.', '_')}.json"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        result_data = {
            'filename': filename,
            'scale_ratio': scale_ratio,
            'detection_results': detection_results,
            'size_calculations': size_calculations,
            'statistics': statistics,
            'analysis_image': analysis_image_path,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析完成: {filename}")
        return jsonify({
            'success': True,
            'results': result_data
        })
        
    except Exception as e:
        logger.error(f"分析错误: {str(e)}")
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

@app.route('/validate_dataset', methods=['POST'])
def validate_dataset():
    """验证数据集"""
    try:
        dataset_path = request.json.get('dataset_path', 'data/dataset')
        validation_results = data_validator.validate_dataset(dataset_path)
        
        return jsonify({
            'success': True,
            'validation_results': validation_results
        })
        
    except Exception as e:
        logger.error(f"数据集验证错误: {str(e)}")
        return jsonify({'error': f'验证失败: {str(e)}'}), 500

@app.route('/results/<filename>')
def get_result(filename):
    """获取分析结果"""
    try:
        result_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if os.path.exists(result_path):
            return send_from_directory(app.config['RESULTS_FOLDER'], filename)
        else:
            return jsonify({'error': '结果文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'获取结果失败: {str(e)}'}), 500

@app.route('/analysis_images/<filename>')
def get_analysis_image(filename):
    """获取分析图片"""
    try:
        analysis_path = os.path.join(app.config['ANALYSIS_FOLDER'], filename)
        if os.path.exists(analysis_path):
            return send_from_directory(app.config['ANALYSIS_FOLDER'], filename)
        else:
            return jsonify({'error': '分析图片不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'获取分析图片失败: {str(e)}'}), 500

def allowed_file(filename):
    """检查文件格式是否支持"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_detection_boxes(image_path, detection_results, original_filename):
    """绘制检测框并保存分析图片"""
    try:
        # 读取原始图片
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图片: {image_path}")
            return None
        
        # 定义颜色映射
        colors = {
            'door': (0, 255, 0),      # 绿色
            'window': (255, 0, 0),    # 蓝色
            'stair': (0, 0, 255),     # 红色
            'elevator': (255, 255, 0), # 青色
            'room': (255, 0, 255),     # 洋红
            'wall': (128, 128, 128),   # 灰色
            'column': (0, 255, 255),  # 黄色
            'bathroom': (128, 0, 128), # 紫色
            'kitchen': (255, 165, 0), # 橙色
            'balcony': (0, 128, 128),  # 深青色
            'corridor': (128, 255, 0)  # 浅绿色
        }
        
        # 绘制检测框
        for detection in detection_results:
            class_name = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0)
            bbox = detection.get('bbox', [])
            
            if len(bbox) >= 4:
                x, y, w, h = bbox[:4]
                color = colors.get(class_name, (255, 255, 255))  # 默认白色
                
                # 绘制边界框
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                
                # 绘制标签
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # 绘制标签背景
                cv2.rectangle(image, (int(x), int(y) - label_size[1] - 10), 
                             (int(x) + label_size[0], int(y)), color, -1)
                
                # 绘制标签文字
                cv2.putText(image, label, (int(x), int(y) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 保存分析图片
        analysis_filename = f"analysis_{original_filename}"
        analysis_path = os.path.join(app.config['ANALYSIS_FOLDER'], analysis_filename)
        cv2.imwrite(analysis_path, image)
        
        logger.info(f"分析图片已保存: {analysis_path}")
        return f"/analysis_images/{analysis_filename}"
        
    except Exception as e:
        logger.error(f"绘制检测框失败: {str(e)}")
        return None

def generate_statistics(detection_results, size_calculations):
    """生成统计信息"""
    statistics = {
        'total_objects': len(detection_results),
        'object_counts': {},
        'total_area': 0,
        'room_count': 0,
        'average_room_size': 0
    }
    
    # 统计对象数量
    for detection in detection_results:
        class_name = detection.get('class', 'unknown')
        statistics['object_counts'][class_name] = statistics['object_counts'].get(class_name, 0) + 1
    
    # 计算总面积和房间信息
    for calc in size_calculations:
        if calc.get('type') == 'room':
            statistics['total_area'] += calc.get('area', 0)
            statistics['room_count'] += 1
    
    if statistics['room_count'] > 0:
        statistics['average_room_size'] = statistics['total_area'] / statistics['room_count']
    
    return statistics

if __name__ == '__main__':
    logger.info("启动建筑平面图分析系统...")
    app.run(debug=True, host='0.0.0.0', port=5000)
