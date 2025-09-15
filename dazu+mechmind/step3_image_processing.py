import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import re

def find_corners(images, pattern_size):
    """检测图像中的角点"""
    world_points = create_world_points(pattern_size)
    obj_points = []
    img_points = []
    used_indices = []
    unused_images = []

    # 确保输出目录存在
    os.makedirs('./processed_images', exist_ok=True)

    for i, fname in tqdm(enumerate(images), desc="Finding corners", total=len(images)):
        img = cv2.imread(fname)
        if img is None:
            print(f"警告：无法读取图片 {fname}")
            unused_images.append(fname)
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret:
            obj_points.append(world_points)
            img_points.append(corners)
            used_indices.append(i)
            # 绘制角点
            cv2.drawChessboardCorners(img, pattern_size, corners.reshape(-1, 1, 2), ret)
            output_fname = f"./processed_images/{i:03d}.png"
            cv2.imwrite(output_fname, img)
        else:
            unused_images.append(fname)

    return obj_points, img_points, used_indices, unused_images

def create_world_points(pattern_size):
    """创建标定板的世界坐标点"""
    width, height = pattern_size
    world_points = np.zeros((width * height, 3), np.float32)
    num = 0
    for i in range(height):
        for j in range(width):
            world_points[num, :2] = [j + 0.5 * (i % 2), i * 0.5]
            num += 1
    return world_points * 0.02

def sort_images(images):
    """按文件名中的数字排序图像"""
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else 0
    return sorted(images, key=extract_number)

def save_processing_results(obj_points, img_points, used_indices, unused_images, img_size):
    """保存图像处理结果"""
    # 保存角点数据
    np.save('./obj_points.npy', obj_points)
    np.save('./img_points.npy', img_points)
    np.save('./used_indices.npy', used_indices)
    np.save('./img_size.npy', img_size)
    
    # 保存处理统计信息
    with open('./processing_summary.txt', 'w', encoding='utf-8') as f:
        f.write("图像处理结果摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"总图像数量: {len(used_indices) + len(unused_images)}\n")
        f.write(f"成功处理图像数量: {len(obj_points)}\n")
        f.write(f"未处理图像数量: {len(unused_images)}\n")
        f.write(f"图像尺寸: {img_size[0]} x {img_size[1]}\n")
        f.write(f"标定板尺寸: 4 x 11\n")
        f.write(f"角点间距: 20mm\n")
        f.write("\n")
        
        if unused_images:
            f.write("未处理的图像:\n")
            for img in unused_images:
                f.write(f"  - {img}\n")
        
        f.write("\n成功处理的图像索引:\n")
        for idx in used_indices:
            f.write(f"  - {idx:03d}.png\n")
    
    print(f"图像处理结果已保存:")
    print(f"  - obj_points.npy: 世界坐标点")
    print(f"  - img_points.npy: 图像坐标点")
    print(f"  - used_indices.npy: 使用的图像索引")
    print(f"  - img_size.npy: 图像尺寸")
    print(f"  - processing_summary.txt: 处理摘要")

def main():
    """主函数：图像处理和角点检测"""
    pattern_size = (4, 11)
    images = glob.glob('./captured_images/*.png')
    
    if not images:
        print("错误：在 ./captured_images/ 目录中未找到图片文件")
        return
        
    images = sort_images(images)
    print(f"找到 {len(images)} 张图片")
    
    # 检测角点
    obj_points, img_points, used_indices, unused_images = find_corners(images, pattern_size)
    
    if len(obj_points) == 0:
        print("错误：没有找到有效的角点，无法进行后续处理")
        return
        
    if unused_images:
        print(f"未处理的图片数量: {len(unused_images)}")
        print(f"未处理的图片: {unused_images}")
    
    print(f"成功处理 {len(obj_points)} 张图片")
    
    # 获取图像尺寸
    img_size = cv2.imread(images[0]).shape[::-1][1:3]
    
    # 保存处理结果
    save_processing_results(obj_points, img_points, used_indices, unused_images, img_size)
    
    print("\n图像处理完成！")
    print("下一步：运行 step4_calibration.py 进行标定计算")

if __name__ == "__main__":
    main()
