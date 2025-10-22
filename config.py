"""
配置文件
系统配置参数
"""

import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# 确保目录存在
for directory in [DATA_DIR, MODELS_DIR, UPLOADS_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Flask配置
class Config:
    """基础配置"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = str(UPLOADS_DIR)
    RESULTS_FOLDER = str(RESULTS_DIR)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # 数据库配置
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f'sqlite:///{BASE_DIR}/app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 日志配置
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = str(LOGS_DIR / 'app.log')
    
    # 模型配置
    MODEL_PATH = str(MODELS_DIR / "yolo_floorplan.pt")
    DEVICE = 'cuda' if os.environ.get('CUDA_AVAILABLE') == 'true' else 'cpu'
    
    # 图像处理配置
    IMAGE_TARGET_SIZE = (640, 640)
    IMAGE_QUALITY = 95
    
    # 检测配置
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 1000
    
    # 比例尺配置
    DEFAULT_SCALE_RATIO = 100
    SUPPORTED_SCALE_RATIOS = [50, 100, 200, 500, 1000]
    
    # 对象类别配置
    OBJECT_CLASSES = [
        'door', 'window', 'stair', 'elevator', 'room',
        'wall', 'column', 'bathroom', 'kitchen', 'balcony', 'corridor'
    ]
    
    # 单位配置
    DEFAULT_UNIT = 'meters'
    SUPPORTED_UNITS = ['mm', 'cm', 'm', 'ft', 'in']
    
    # API配置
    API_RATE_LIMIT = '100 per hour'
    CORS_ORIGINS = ['*']
    
    # 缓存配置
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    TESTING = False
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    TESTING = False
    LOG_LEVEL = 'WARNING'
    
    # 生产环境安全配置
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("生产环境必须设置SECRET_KEY环境变量")

class TestingConfig(Config):
    """测试环境配置"""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = 'DEBUG'
    
    # 测试数据库
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{BASE_DIR}/test.db'
    
    # 测试文件路径
    UPLOAD_FOLDER = str(BASE_DIR / "test_uploads")
    RESULTS_FOLDER = str(BASE_DIR / "test_results")

# 配置字典
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# 获取当前配置
def get_config():
    """获取当前配置"""
    config_name = os.environ.get('FLASK_ENV', 'development')
    return config.get(config_name, config['default'])

# 模型配置
MODEL_CONFIG = {
    'yolo': {
        'model_path': str(MODELS_DIR / "yolo_floorplan.pt"),
        'input_size': (640, 640),
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'max_detections': 1000
    },
    'classification': {
        'model_path': str(MODELS_DIR / "classifier.pt"),
        'input_size': (224, 224),
        'num_classes': 11
    }
}

# 数据集配置
DATASET_CONFIG = {
    'name': 'floorplan_dataset',
    'version': '1.0',
    'description': '建筑平面图数据集',
    'classes': [
        'door', 'window', 'stair', 'elevator', 'room',
        'wall', 'column', 'bathroom', 'kitchen', 'balcony', 'corridor'
    ],
    'image_formats': ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'],
    'annotation_format': 'yolo',
    'train_ratio': 0.7,
    'val_ratio': 0.2,
    'test_ratio': 0.1
}

# 验证规则
VALIDATION_RULES = {
    'image': {
        'min_width': 100,
        'min_height': 100,
        'max_width': 10000,
        'max_height': 10000,
        'max_size_mb': 16
    },
    'annotation': {
        'min_objects': 1,
        'max_objects': 1000,
        'min_confidence': 0.1,
        'max_confidence': 1.0
    },
    'detection': {
        'min_confidence': 0.3,
        'max_objects': 100,
        'min_area': 100,
        'max_area': 100000
    }
}

# 日志配置
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        },
        'detailed': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s.%(funcName)s:%(lineno)d: %(message)s',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'default',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / 'app.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        'app': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}
