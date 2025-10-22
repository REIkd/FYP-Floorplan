"""
启动脚本
启动建筑平面图分析系统
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/app.log', encoding='utf-8')
        ]
    )

def check_dependencies():
    """检查依赖"""
    required_packages = [
        'torch', 'torchvision', 'opencv-python', 'numpy', 
        'Pillow', 'matplotlib', 'flask', 'flask-cors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def check_directories():
    """检查目录结构"""
    required_dirs = [
        'data', 'models', 'uploads', 'results', 'logs',
        'templates', 'static', 'tests'
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print("❌ 缺少以下目录:")
        for dir_name in missing_dirs:
            print(f"  - {dir_name}")
        return False
    
    print("✅ 目录结构完整")
    return True

def check_models():
    """检查模型文件"""
    model_files = [
        'models/yolo_floorplan.pt'
    ]
    
    missing_models = []
    
    for model_file in model_files:
        if not Path(model_file).exists():
            missing_models.append(model_file)
    
    if missing_models:
        print("⚠️  缺少以下模型文件:")
        for model_file in missing_models:
            print(f"  - {model_file}")
        print("系统将使用模拟检测模式")
    else:
        print("✅ 模型文件完整")
    
    return True

def run_tests():
    """运行测试"""
    print("\n🧪 运行系统测试...")
    
    try:
        from run_tests import run_all_tests
        success = run_all_tests()
        
        if success:
            print("✅ 所有测试通过")
            return True
        else:
            print("❌ 部分测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 测试运行失败: {str(e)}")
        return False

def start_application():
    """启动应用"""
    print("\n🚀 启动建筑平面图分析系统...")
    
    try:
        from app import app
        from config import get_config
        
        # 获取配置
        config = get_config()
        app.config.from_object(config)
        
        # 启动应用
        print(f"🌐 应用将在 http://localhost:5000 启动")
        print("📱 使用浏览器访问Web界面")
        print("⏹️  按 Ctrl+C 停止应用")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=config.DEBUG
        )
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 应用启动失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🏗️  建筑平面图分析系统")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    
    # 检查系统
    print("\n🔍 检查系统环境...")
    
    if not check_dependencies():
        return 1
    
    if not check_directories():
        return 1
    
    check_models()
    
    # 运行测试
    if '--test' in sys.argv:
        if not run_tests():
            return 1
    
    # 启动应用
    if '--no-start' not in sys.argv:
        start_application()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
