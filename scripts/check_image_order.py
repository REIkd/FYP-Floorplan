#!/usr/bin/env python3
"""
检查图片顺序和增强方式

帮助确认图片是否按照预期的顺序排列：
- 第 1-101 张：原始图片
- 第 102-202 张：镜像图片
- 第 203-303 张：旋转图片
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path


def check_if_flipped(img1, img2):
    """
    检查两张图片是否为镜像关系
    
    方法：将img2水平翻转后与img1比较相似度
    """
    if img1.shape != img2.shape:
        return False, 0.0
    
    # 水平翻转img2
    img2_flipped = cv2.flip(img2, 1)
    
    # 计算相似度（使用结构相似度）
    # 简化版：计算像素差异
    diff = np.abs(img1.astype(float) - img2_flipped.astype(float)).mean()
    similarity = 1.0 - (diff / 255.0)
    
    return similarity > 0.95, similarity


def check_if_rotated(img1, img2):
    """
    检查两张图片是否为旋转90度关系
    
    方法：将img2逆时针旋转90度后与img1比较
    """
    # 注意：旋转后尺寸会改变
    # 逆时针旋转90度 = 顺时针旋转270度
    img2_rotated = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # 检查尺寸是否匹配（旋转后宽高互换）
    if img1.shape[:2] != img2_rotated.shape[:2]:
        # 尝试调整大小
        img2_rotated = cv2.resize(img2_rotated, (img1.shape[1], img1.shape[0]))
    
    # 计算相似度
    diff = np.abs(img1.astype(float) - img2_rotated.astype(float)).mean()
    similarity = 1.0 - (diff / 255.0)
    
    return similarity > 0.95, similarity


def analyze_image_order(images_dir):
    """分析图片顺序和增强关系"""
    
    print("=" * 80)
    print("图片顺序和增强方式检查工具")
    print("=" * 80)
    print()
    
    # 获取所有图片
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    
    total = len(image_files)
    print(f"找到图片总数: {total}")
    
    if total == 0:
        print("错误: 没有找到任何图片！")
        return
    
    if total != 303:
        print(f"⚠️  警告: 期望303张图片，实际找到{total}张")
        print()
    
    # 计算每组数量
    num_original = total // 3
    
    print(f"假设分组:")
    print(f"  原始图片: 第 1-{num_original} 张")
    print(f"  镜像图片: 第 {num_original+1}-{num_original*2} 张")
    print(f"  旋转图片: 第 {num_original*2+1}-{total} 张")
    print()
    
    # 显示示例文件名
    print("=" * 80)
    print("文件名示例")
    print("=" * 80)
    print(f"原始图片示例 (第1张): {Path(image_files[0]).name}")
    if num_original < total:
        print(f"镜像图片示例 (第{num_original+1}张): {Path(image_files[num_original]).name}")
    if num_original * 2 < total:
        print(f"旋转图片示例 (第{num_original*2+1}张): {Path(image_files[num_original*2]).name}")
    print()
    
    # 抽样验证增强关系
    print("=" * 80)
    print("验证增强关系 (抽样检查)")
    print("=" * 80)
    print("正在加载图片并比对...")
    print()
    
    # 检查前3对作为示例
    samples_to_check = min(3, num_original)
    
    for i in range(samples_to_check):
        original_idx = i
        flipped_idx = i + num_original
        rotated_idx = i + num_original * 2
        
        if rotated_idx >= total:
            break
        
        print(f"检查组 {i+1}:")
        print(f"  原始: {Path(image_files[original_idx]).name}")
        
        # 读取图片
        try:
            img_original = cv2.imread(image_files[original_idx])
            img_flipped = cv2.imread(image_files[flipped_idx])
            img_rotated = cv2.imread(image_files[rotated_idx])
            
            if img_original is None or img_flipped is None or img_rotated is None:
                print("  ⚠️  无法读取图片")
                continue
            
            # 检查镜像关系
            print(f"  镜像: {Path(image_files[flipped_idx]).name}")
            is_flip, flip_sim = check_if_flipped(img_original, img_flipped)
            if is_flip:
                print(f"    ✓ 确认为镜像关系 (相似度: {flip_sim:.3f})")
            else:
                print(f"    ✗ 不是镜像关系 (相似度: {flip_sim:.3f})")
                print(f"    ⚠️  这可能表示图片顺序不正确！")
            
            # 检查旋转关系
            print(f"  旋转: {Path(image_files[rotated_idx]).name}")
            is_rot, rot_sim = check_if_rotated(img_original, img_rotated)
            if is_rot:
                print(f"    ✓ 确认为旋转90度关系 (相似度: {rot_sim:.3f})")
            else:
                print(f"    ✗ 不是旋转90度关系 (相似度: {rot_sim:.3f})")
                print(f"    ⚠️  这可能表示图片顺序不正确！")
            
            print()
            
        except Exception as e:
            print(f"  ✗ 检查失败: {e}")
            print()
    
    print("=" * 80)
    print("总结")
    print("=" * 80)
    print()
    print("如果上述检查显示:")
    print("  ✓ 镜像和旋转关系都正确 → 可以安全使用自动标注生成脚本")
    print("  ✗ 关系不正确 → 需要重新整理图片顺序或修改脚本逻辑")
    print()
    print("提示:")
    print("  1. 如果图片顺序不对，请重新组织图片文件")
    print("  2. 确保文件名按预期排序（可使用编号前缀）")
    print("  3. 如果增强方式不同（如右旋转90度），需要修改脚本的变换函数")


def list_all_images(images_dir, output_file=None):
    """列出所有图片及其索引"""
    
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    
    print("=" * 80)
    print("所有图片列表")
    print("=" * 80)
    
    for i, img_path in enumerate(image_files, 1):
        line = f"{i:3d}. {Path(img_path).name}"
        print(line)
        
        if output_file:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
    
    if output_file:
        print()
        print(f"列表已保存到: {output_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='检查图片顺序和增强方式')
    parser.add_argument('--images-dir', type=str, default='data/images',
                        help='图片目录路径')
    parser.add_argument('--list-all', action='store_true',
                        help='列出所有图片及其索引')
    parser.add_argument('--output', type=str,
                        help='输出列表到文件')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.images_dir):
        print(f"错误: 图片目录不存在: {args.images_dir}")
        return
    
    if args.list_all:
        list_all_images(args.images_dir, args.output)
    else:
        analyze_image_order(args.images_dir)


if __name__ == '__main__':
    main()

