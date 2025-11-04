#!/usr/bin/env python3
"""
自动生成增强数据的标注文件
通过坐标变换从原始标注生成镜像和旋转后的标注

使用方法：
    python scripts/auto_generate_labels.py

功能：
    1. 识别原始的101张图片及其标注
    2. 自动为镜像（水平翻转）的图片生成标注
    3. 自动为旋转90度的图片生成标注
"""

import os
import glob
from pathlib import Path
import shutil


def parse_yolo_line(line):
    """解析YOLO格式的一行标注"""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    
    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])
    
    return class_id, x_center, y_center, width, height


def format_yolo_line(class_id, x_center, y_center, width, height):
    """格式化为YOLO格式的一行"""
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def transform_horizontal_flip(class_id, x_center, y_center, width, height):
    """
    水平翻转（镜像）的坐标变换
    
    原理：
        - x坐标翻转：new_x = 1 - x
        - y坐标不变
        - 宽高不变
    """
    new_x = 1.0 - x_center
    new_y = y_center
    new_w = width
    new_h = height
    
    return class_id, new_x, new_y, new_w, new_h


def transform_rotate_left_90(class_id, x_center, y_center, width, height):
    """
    向左旋转90度（逆时针）的坐标变换
    
    原理：
        图像向左旋转90度后：
        - 原图的顶部变成新图的左侧
        - 原图的左侧变成新图的底部
        
        坐标变换：
        - new_x = y_center
        - new_y = 1 - x_center
        - new_width = height
        - new_height = width
    """
    new_x = y_center
    new_y = 1.0 - x_center
    new_w = height
    new_h = width
    
    return class_id, new_x, new_y, new_w, new_h


def identify_image_type(filename):
    """
    识别图片类型
    
    返回：
        'original' - 原始图片
        'flipped' - 镜像图片
        'rotated' - 旋转图片
        None - 无法识别
    """
    # 根据文件名模式识别
    # 这里需要根据你的实际文件命名规则调整
    
    # 示例模式（请根据实际情况修改）：
    # 原始: FloorPlan-1-_JPG.rf.xxxxx.jpg
    # 镜像: FloorPlan-1-_JPG.rf.xxxxx_flip.jpg 或其他标识
    # 旋转: FloorPlan-1-_JPG.rf.xxxxx_rot90.jpg 或其他标识
    
    name_lower = filename.lower()
    
    if 'flip' in name_lower or 'mirror' in name_lower:
        return 'flipped'
    elif 'rot' in name_lower or 'rotate' in name_lower:
        return 'rotated'
    else:
        return 'original'


def find_original_label(image_path, labels_dir):
    """
    根据增强图片找到对应的原始标注文件
    
    参数：
        image_path: 增强图片的路径
        labels_dir: 标注目录
    
    返回：
        原始标注文件路径，如果找不到返回None
    """
    image_name = Path(image_path).stem
    
    # 尝试移除常见的增强标识
    # 根据实际命名规则调整
    original_name = image_name
    
    # 移除常见后缀
    for suffix in ['_flip', '_flipped', '_mirror', '_rot90', '_rotated', '_rot']:
        if suffix in original_name:
            original_name = original_name.replace(suffix, '')
            break
    
    # 查找对应的标注文件
    label_path = os.path.join(labels_dir, original_name + '.txt')
    
    if os.path.exists(label_path):
        return label_path
    
    return None


def generate_label_from_original(original_label_path, output_path, transform_type):
    """
    从原始标注生成变换后的标注
    
    参数：
        original_label_path: 原始标注文件路径
        output_path: 输出标注文件路径
        transform_type: 变换类型 ('flipped' 或 'rotated')
    """
    if not os.path.exists(original_label_path):
        print(f"警告: 原始标注文件不存在: {original_label_path}")
        return False
    
    # 选择变换函数
    if transform_type == 'flipped':
        transform_func = transform_horizontal_flip
    elif transform_type == 'rotated':
        transform_func = transform_rotate_left_90
    else:
        print(f"错误: 未知的变换类型: {transform_type}")
        return False
    
    # 读取原始标注并变换
    transformed_lines = []
    
    try:
        with open(original_label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parsed = parse_yolo_line(line)
                if parsed is None:
                    continue
                
                # 应用坐标变换
                transformed = transform_func(*parsed)
                transformed_line = format_yolo_line(*transformed)
                transformed_lines.append(transformed_line)
        
        # 写入新标注文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(transformed_lines)
        
        return True
        
    except Exception as e:
        print(f"错误: 处理标注文件时出错: {e}")
        return False


def auto_generate_labels_simple(images_dir, labels_dir):
    """
    简化版：根据文件名模式自动生成标注
    
    假设文件命名规则：
    - 原始图片的编号是 1-101
    - 镜像图片的编号是 102-202
    - 旋转图片的编号是 203-303
    """
    print("=" * 80)
    print("自动标注生成工具 - 简化模式")
    print("=" * 80)
    print("\n假设数据组织方式：")
    print("  - 图片 1-101: 原始图片")
    print("  - 图片 102-202: 镜像图片（对应原始图片 1-101）")
    print("  - 图片 203-303: 旋转图片（对应原始图片 1-101）")
    print()
    
    # 获取所有图片文件
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    print(f"找到图片总数: {len(image_files)}")
    
    if len(image_files) != 303:
        print(f"警告: 期望303张图片，实际找到{len(image_files)}张")
    
    # 构建图片列表（假设按文件名排序后顺序为：原始→镜像→旋转）
    total = len(image_files)
    num_original = total // 3
    
    print(f"原始图片数量: {num_original}")
    print(f"每组增强图片数量: {num_original}")
    
    generated_count = 0
    skipped_count = 0
    
    # 生成镜像图片的标注（第102-202张）
    print("\n" + "=" * 80)
    print("步骤 1: 生成镜像图片的标注")
    print("=" * 80)
    
    for i in range(num_original):
        original_idx = i
        flipped_idx = i + num_original
        
        if flipped_idx >= len(image_files):
            break
        
        original_image = image_files[original_idx]
        flipped_image = image_files[flipped_idx]
        
        original_label = os.path.join(labels_dir, Path(original_image).stem + '.txt')
        flipped_label = os.path.join(labels_dir, Path(flipped_image).stem + '.txt')
        
        # 如果原始标注不存在，跳过
        if not os.path.exists(original_label):
            print(f"跳过 (无原始标注): {Path(original_image).name} -> {Path(flipped_image).name}")
            skipped_count += 1
            continue
        
        # 如果目标标注已存在，跳过
        if os.path.exists(flipped_label):
            print(f"已存在: {Path(flipped_label).name}")
            continue
        
        # 生成镜像标注
        success = generate_label_from_original(original_label, flipped_label, 'flipped')
        
        if success:
            print(f"✓ 生成: {Path(flipped_image).name} <- {Path(original_image).name}")
            generated_count += 1
        else:
            print(f"✗ 失败: {Path(flipped_image).name}")
    
    # 生成旋转图片的标注（第203-303张）
    print("\n" + "=" * 80)
    print("步骤 2: 生成旋转图片的标注")
    print("=" * 80)
    
    for i in range(num_original):
        original_idx = i
        rotated_idx = i + num_original * 2
        
        if rotated_idx >= len(image_files):
            break
        
        original_image = image_files[original_idx]
        rotated_image = image_files[rotated_idx]
        
        original_label = os.path.join(labels_dir, Path(original_image).stem + '.txt')
        rotated_label = os.path.join(labels_dir, Path(rotated_image).stem + '.txt')
        
        # 如果原始标注不存在，跳过
        if not os.path.exists(original_label):
            print(f"跳过 (无原始标注): {Path(original_image).name} -> {Path(rotated_image).name}")
            skipped_count += 1
            continue
        
        # 如果目标标注已存在，跳过
        if os.path.exists(rotated_label):
            print(f"已存在: {Path(rotated_label).name}")
            continue
        
        # 生成旋转标注
        success = generate_label_from_original(original_label, rotated_label, 'rotated')
        
        if success:
            print(f"✓ 生成: {Path(rotated_image).name} <- {Path(original_image).name}")
            generated_count += 1
        else:
            print(f"✗ 失败: {Path(rotated_image).name}")
    
    # 统计结果
    print("\n" + "=" * 80)
    print("生成完成！")
    print("=" * 80)
    print(f"成功生成: {generated_count} 个标注文件")
    print(f"跳过: {skipped_count} 个（原始标注不存在）")
    print()
    print("提示：")
    print("  1. 请验证生成的标注是否正确")
    print("  2. 可以使用 scripts/visualize_dataset.py 查看标注效果")
    print("  3. 每次标注完原始图片后，重新运行此脚本更新增强图片的标注")


def test_transforms():
    """测试坐标变换是否正确"""
    print("=" * 80)
    print("坐标变换测试")
    print("=" * 80)
    
    # 测试用例：图片中心的一个正方形框
    test_cases = [
        ("中心正方形", 0, 0.5, 0.5, 0.2, 0.2),
        ("左上角", 0, 0.2, 0.2, 0.1, 0.1),
        ("右下角", 0, 0.8, 0.8, 0.1, 0.1),
    ]
    
    for name, class_id, x, y, w, h in test_cases:
        print(f"\n测试: {name}")
        print(f"  原始坐标: class={class_id}, x={x}, y={y}, w={w}, h={h}")
        
        # 水平翻转
        flipped = transform_horizontal_flip(class_id, x, y, w, h)
        print(f"  水平翻转: x={flipped[1]:.3f}, y={flipped[2]:.3f}, w={flipped[3]:.3f}, h={flipped[4]:.3f}")
        
        # 旋转90度
        rotated = transform_rotate_left_90(class_id, x, y, w, h)
        print(f"  旋转90度: x={rotated[1]:.3f}, y={rotated[2]:.3f}, w={rotated[3]:.3f}, h={rotated[4]:.3f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='自动生成增强数据的标注文件')
    parser.add_argument('--images-dir', type=str, default='data/images',
                        help='图片目录路径')
    parser.add_argument('--labels-dir', type=str, default='data/labels_detection',
                        help='标注目录路径')
    parser.add_argument('--test', action='store_true',
                        help='运行坐标变换测试')
    
    args = parser.parse_args()
    
    # 如果是测试模式
    if args.test:
        test_transforms()
        return
    
    # 检查目录
    if not os.path.exists(args.images_dir):
        print(f"错误: 图片目录不存在: {args.images_dir}")
        return
    
    if not os.path.exists(args.labels_dir):
        print(f"错误: 标注目录不存在: {args.labels_dir}")
        return
    
    # 执行自动生成
    print(f"图片目录: {args.images_dir}")
    print(f"标注目录: {args.labels_dir}")
    print()
    
    # 询问确认
    response = input("是否继续？这将根据原始标注自动生成增强图片的标注 (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("已取消")
        return
    
    print()
    auto_generate_labels_simple(args.images_dir, args.labels_dir)


if __name__ == '__main__':
    main()

