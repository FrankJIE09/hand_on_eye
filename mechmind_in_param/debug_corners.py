#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试角点检测脚本
用于分析图像和测试不同的检测方法
"""

import cv2
import numpy as np
import glob
import os

def analyze_image(image_path):
    """分析单张图像"""
    print(f"\n=== 分析图像: {image_path} ===")
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像")
        return
    
    print(f"图像尺寸: {img.shape}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 分析图像统计信息
    print(f"灰度值范围: {gray.min()} - {gray.max()}")
    print(f"平均灰度值: {gray.mean():.2f}")
    print(f"标准差: {gray.std():.2f}")
    
    # 检查图像是否有足够的对比度
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    non_zero_bins = np.count_nonzero(hist)
    print(f"非零灰度级数量: {non_zero_bins}")
    
    # 尝试不同的预处理方法
    methods = {
        'original': gray,
        'equalized': cv2.equalizeHist(gray),
        'clahe': cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray),
        'threshold': cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        'adaptive_thresh': cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    }
    
    # 测试不同的标定板尺寸
    pattern_sizes = [(5, 4), (4, 5), (4, 4), (5, 5)]
    
    for method_name, processed_img in methods.items():
        print(f"\n--- 测试方法: {method_name} ---")
        
        for pattern_size in pattern_sizes:
            # 尝试检测圆点网格
            ret, corners = cv2.findCirclesGrid(processed_img, pattern_size, 
                                            flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            if ret:
                print(f"  ✓ 找到 {pattern_size} 圆点网格，角点数量: {len(corners)}")
                
                # 保存检测结果
                result_img = img.copy()
                cv2.drawChessboardCorners(result_img, pattern_size, corners, ret)
                
                output_name = f"debug_{os.path.basename(image_path)}_{method_name}_{pattern_size[0]}x{pattern_size[1]}.png"
                cv2.imwrite(output_name, result_img)
                print(f"  ✓ 结果已保存: {output_name}")
                return True
            else:
                print(f"  ✗ 未找到 {pattern_size} 圆点网格")
        
        # 尝试检测棋盘格（作为备选）
        for pattern_size in pattern_sizes:
            ret, corners = cv2.findChessboardCorners(processed_img, pattern_size)
            if ret:
                print(f"  ✓ 找到 {pattern_size} 棋盘格，角点数量: {len(corners)}")
                return True
    
    return False

def main():
    """主函数"""
    print("=== 角点检测调试工具 ===")
    
    # 查找图像文件
    images = glob.glob('./captured_images/*.png')
    if not images:
        print("未找到图像文件")
        return
    
    print(f"找到 {len(images)} 张图像")
    
    success_count = 0
    for image_path in images:
        if analyze_image(image_path):
            success_count += 1
    
    print(f"\n=== 总结 ===")
    print(f"成功检测: {success_count}/{len(images)} 张图像")
    
    if success_count == 0:
        print("\n建议:")
        print("1. 检查标定板是否正确放置")
        print("2. 调整光照条件，避免过曝")
        print("3. 确保标定板完整可见")
        print("4. 检查标定板规格是否正确")

if __name__ == "__main__":
    main()
