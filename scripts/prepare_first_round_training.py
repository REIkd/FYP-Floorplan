#!/usr/bin/env python3
"""
准备第一轮训练数据

从已标注的53张图片生成增强数据集用于第一轮训练
"""

import os
import glob
import shutil
import cv2
import numpy as np
from pathlib import Path
import random


def extract_floorplan_number(filename):
    """从文件名中提取 FloorPlan 编号"""
    try:
        name = os.path.basename(filename)
        if name.startswith('FloorPlan-'):
            parts = name.split('-')
            if len(parts) >= 2:
                number = int(parts[1])
                return number
    except:
        pass
    return None


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
    """水平翻转的坐标变换"""
    new_x = 1.0 - x_center
    new_y = y_center
    new_w = width
    new_h = height
    return class_id, new_x, new_y, new_w, new_h


def transform_rotate_left_90(class_id, x_center, y_center, width, height):
    """向左旋转90度的坐标变换"""
    new_x = y_center
    new_y = 1.0 - x_center
    new_w = height
    new_h = width
    return class_id, new_x, new_y, new_w, new_h


def flip_image_horizontal(image):
    """水平翻转图片"""
    return cv2.flip(image, 1)


def rotate_image_left_90(image):
    """逆时针旋转图片90度"""
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def augment_single_image(image_path, label_path, output_dir_images, output_dir_labels, base_name):
    """为单张图片生成增强数据"""
    
    # 读取原始图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"  无法读取图片: {image_path}")
        return 0
    
    # 读取标注
    if not os.path.exists(label_path):
        print(f"  标注不存在: {label_path}")
        return 0
    
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    count = 0
    
    # 1. 保存原始图片和标注
    original_img_path = os.path.join(output_dir_images, f"{base_name}_original.jpg")
    original_label_path = os.path.join(output_dir_labels, f"{base_name}_original.txt")
    cv2.imwrite(original_img_path, image)
    shutil.copy2(label_path, original_label_path)
    count += 1
    
    # 2. 生成水平翻转
    flipped_img = flip_image_horizontal(image)
    flipped_img_path = os.path.join(output_dir_images, f"{base_name}_flip.jpg")
    cv2.imwrite(flipped_img_path, flipped_img)
    
    flipped_label_path = os.path.join(output_dir_labels, f"{base_name}_flip.txt")
    with open(flipped_label_path, 'w', encoding='utf-8') as f:
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parsed = parse_yolo_line(line)
            if parsed:
                transformed = transform_horizontal_flip(*parsed)
                f.write(format_yolo_line(*transformed))
    count += 1
    
    # 3. 生成旋转90度
    rotated_img = rotate_image_left_90(image)
    rotated_img_path = os.path.join(output_dir_images, f"{base_name}_rot90.jpg")
    cv2.imwrite(rotated_img_path, rotated_img)
    
    rotated_label_path = os.path.join(output_dir_labels, f"{base_name}_rot90.txt")
    with open(rotated_label_path, 'w', encoding='utf-8') as f:
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parsed = parse_yolo_line(line)
            if parsed:
                transformed = transform_rotate_left_90(*parsed)
                f.write(format_yolo_line(*transformed))
    count += 1
    
    return count


def prepare_first_round_dataset(
    images_source_dir='data/images',
    labels_source_dir='data/labels_detection',
    output_base_dir='data/train_first_round',
    max_number=53
):
    """
    准备第一轮训练数据集
    """
    
    print("=" * 80)
    print("第一轮训练数据准备")
    print("=" * 80)
    print()
    print(f"源图片目录: {images_source_dir}")
    print(f"源标注目录: {labels_source_dir}")
    print(f"输出目录: {output_base_dir}")
    print(f"使用图片: FloorPlan 1-{max_number}")
    print()
    
    # 创建输出目录
    output_images_dir = os.path.join(output_base_dir, 'images')
    output_labels_dir = os.path.join(output_base_dir, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # 获取所有图片并分组
    image_files = glob.glob(os.path.join(images_source_dir, "*.jpg"))
    image_files.extend(glob.glob(os.path.join(images_source_dir, "*.JPG")))
    
    from collections import defaultdict
    groups = defaultdict(list)
    
    for img_path in image_files:
        number = extract_floorplan_number(img_path)
        if number is not None and 1 <= number <= max_number:
            groups[number].append(img_path)
    
    for number in groups:
        groups[number].sort()
    
    print(f"找到 {len(groups)} 个 FloorPlan 组（1-{max_number}）")
    print()
    print("=" * 80)
    print("生成增强数据")
    print("=" * 80)
    print()
    
    total_images = 0
    processed = 0
    
    for number in range(1, max_number + 1):
        if number not in groups or len(groups[number]) == 0:
            print(f"  FloorPlan-{number:2d}: 未找到图片")
            continue
        
        # 取第一张作为原始图片
        original_image = groups[number][0]
        original_name = Path(original_image).stem
        original_label = os.path.join(labels_source_dir, original_name + '.txt')
        
        if not os.path.exists(original_label):
            print(f"  FloorPlan-{number:2d}: 缺少标注文件")
            continue
        
        print(f"  FloorPlan-{number:2d}: {Path(original_image).name}")
        
        # 生成增强数据
        count = augment_single_image(
            original_image,
            original_label,
            output_images_dir,
            output_labels_dir,
            f"floorplan_{number:03d}"
        )
        
        if count > 0:
            total_images += count
            processed += 1
            print(f"    生成 {count} 张增强图片")
    
    # 生成数据集划分文件
    print()
    print("=" * 80)
    print("生成数据集划分")
    print("=" * 80)
    print()
    
    # 获取所有生成的图片
    all_images = sorted(glob.glob(os.path.join(output_images_dir, "*.jpg")))
    all_images = [Path(img).stem for img in all_images]
    
    # 划分训练集和验证集 (80% 训练, 20% 验证)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # 保存划分文件
    splits_dir = os.path.join(output_base_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    with open(os.path.join(splits_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for img_name in train_images:
            f.write(f"images/{img_name}.jpg\n")
    
    with open(os.path.join(splits_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for img_name in val_images:
            f.write(f"images/{img_name}.jpg\n")
    
    # 复制 classes.txt
    shutil.copy2(
        os.path.join(labels_source_dir, 'classes.txt'),
        os.path.join(output_labels_dir, 'classes.txt')
    )
    
    print(f"训练集: {len(train_images)} 张")
    print(f"验证集: {len(val_images)} 张")
    print()
    
    # 生成配置文件
    config_content = f"""# 第一轮训练配置文件
# 数据集路径
path: {os.path.abspath(output_base_dir)}
train: splits/train.txt
val: splits/val.txt

# 类别定义
names:
  0: door
  1: window
  2: table
  3: chair
  4: bed
  5: sofa
  6: toilet
  7: sink
  8: bathtub
  9: stove
  10: refrigerator
  11: wardrobe
  12: tv
  13: desk
  14: washingmachine
  15: loadbearing_wall
  16: aircondition
  17: cupboard

nc: 18

# 训练参数（第一轮：快速验证）
epochs: 50
batch: 8
imgsz: 640
"""
    
    config_path = os.path.join(output_base_dir, 'dataset.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("=" * 80)
    print("准备完成！")
    print("=" * 80)
    print()
    print("统计信息:")
    print(f"  处理的原始图片: {processed} 张")
    print(f"  生成的总图片数: {total_images} 张")
    print(f"  训练集: {len(train_images)} 张")
    print(f"  验证集: {len(val_images)} 张")
    print()
    print("生成的文件:")
    print(f"  图片目录: {output_images_dir}")
    print(f"  标注目录: {output_labels_dir}")
    print(f"  配置文件: {config_path}")
    print()
    print("下一步：开始训练")
    print(f"  yolo train model=yolov8n.pt data={config_path} epochs=50")
    

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='准备第一轮训练数据')
    parser.add_argument('--images-dir', type=str, default='data/images',
                        help='源图片目录')
    parser.add_argument('--labels-dir', type=str, default='data/labels_detection',
                        help='源标注目录')
    parser.add_argument('--output-dir', type=str, default='data/train_first_round',
                        help='输出目录')
    parser.add_argument('--max-number', type=int, default=53,
                        help='最大 FloorPlan 编号')
    
    args = parser.parse_args()
    
    prepare_first_round_dataset(
        args.images_dir,
        args.labels_dir,
        args.output_dir,
        args.max_number
    )


if __name__ == '__main__':
    main()

