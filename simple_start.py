"""
简化启动脚本
跳过依赖检查，直接启动应用
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """设置日志"""
    # 确保logs目录存在
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def start_application():
    """启动应用"""
    print("启动建筑平面图分析系统...")
    
    try:
        from app import app
        from config import get_config
        
        # 获取配置
        config = get_config()
        app.config.from_object(config)
        
        # 启动应用
        print("应用将在 http://localhost:5000 启动")
        print("使用浏览器访问Web界面")
        print("按 Ctrl+C 停止应用")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=config.DEBUG
        )
        
    except KeyboardInterrupt:
        print("\n应用已停止")
    except Exception as e:
        print(f"应用启动失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("建筑平面图分析系统")
    print("=" * 60)
    
    # 设置日志
    setup_logging()
    
    # 直接启动应用
    start_application()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
