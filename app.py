"""
建筑平面图分析系统 - 主应用入口
基于深度学习的平面图识别和分析系统
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from datetime import datetime
import logging

# 导入自定义模块
from models.floorplan_detector import FloorplanDetector
from utils.image_processor import ImageProcessor
from utils.scale_calculator import ScaleCalculator
from utils.data_validator import DataValidator

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 配置
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 确保必要的文件夹存在
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# 初始化组件
detector = FloorplanDetector()
image_processor = ImageProcessor()
scale_calculator = ScaleCalculator()
data_validator = DataValidator()

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
        processed_image = image_processor.preprocess_image(filepath)
        
        # 检测对象
        detection_results = detector.detect_objects(processed_image)
        
        # 计算尺寸
        size_calculations = scale_calculator.calculate_sizes(
            detection_results, scale_ratio
        )
        
        # 生成统计报告
        statistics = generate_statistics(detection_results, size_calculations)
        
        # 保存结果
        result_filename = f"result_{filename.replace('.', '_')}.json"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        result_data = {
            'filename': filename,
            'scale_ratio': scale_ratio,
            'detection_results': detection_results,
            'size_calculations': size_calculations,
            'statistics': statistics,
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

def allowed_file(filename):
    """检查文件格式是否支持"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
