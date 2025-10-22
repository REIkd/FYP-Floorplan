"""
运行测试脚本
执行所有测试并生成报告
"""

import unittest
import sys
import os
from io import StringIO

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("建筑平面图分析系统 - 测试套件")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加模型测试
    from tests.test_models import TestFloorplanDetector, TestYOLODetector, TestObjectClassifier
    test_suite.addTest(unittest.makeSuite(TestFloorplanDetector))
    test_suite.addTest(unittest.makeSuite(TestYOLODetector))
    test_suite.addTest(unittest.makeSuite(TestObjectClassifier))
    
    # 添加工具测试
    from tests.test_utils import TestImageProcessor, TestScaleCalculator, TestDataValidator, TestFileUtils
    test_suite.addTest(unittest.makeSuite(TestImageProcessor))
    test_suite.addTest(unittest.makeSuite(TestScaleCalculator))
    test_suite.addTest(unittest.makeSuite(TestDataValidator))
    test_suite.addTest(unittest.makeSuite(TestFileUtils))
    
    # 添加集成测试
    from tests.test_integration import TestIntegration
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # 计算成功率
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n成功率: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("✅ 测试通过！系统运行良好。")
    elif success_rate >= 70:
        print("⚠️  测试基本通过，但有一些问题需要关注。")
    else:
        print("❌ 测试失败较多，需要修复问题。")
    
    return result.wasSuccessful()

def run_specific_tests(test_module):
    """运行特定测试模块"""
    print(f"运行 {test_module} 测试...")
    
    if test_module == 'models':
        from tests.test_models import TestFloorplanDetector, TestYOLODetector, TestObjectClassifier
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestFloorplanDetector))
        suite.addTest(unittest.makeSuite(TestYOLODetector))
        suite.addTest(unittest.makeSuite(TestObjectClassifier))
    elif test_module == 'utils':
        from tests.test_utils import TestImageProcessor, TestScaleCalculator, TestDataValidator, TestFileUtils
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestImageProcessor))
        suite.addTest(unittest.makeSuite(TestScaleCalculator))
        suite.addTest(unittest.makeSuite(TestDataValidator))
        suite.addTest(unittest.makeSuite(TestFileUtils))
    elif test_module == 'integration':
        from tests.test_integration import TestIntegration
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestIntegration))
    else:
        print(f"未知的测试模块: {test_module}")
        return False
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='运行建筑平面图分析系统测试')
    parser.add_argument('--module', choices=['models', 'utils', 'integration', 'all'], 
                       default='all', help='要运行的测试模块')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.module == 'all':
        success = run_all_tests()
    else:
        success = run_specific_tests(args.module)
    
    sys.exit(0 if success else 1)
