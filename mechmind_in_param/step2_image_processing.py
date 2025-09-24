import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
import re
import pandas as pd

def preprocess_image(img):
    """图像预处理，改善光照不均问题"""
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 方法1：直方图均衡化
    equalized = cv2.equalizeHist(gray)
    
    # 方法2：CLAHE (对比度限制自适应直方图均衡化)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    
    # 方法3：高斯滤波去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 方法4：双边滤波保持边缘
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    return {
        'original': gray,
        'equalized': equalized,
        'clahe': clahe_img,
        'blurred': blurred,
        'bilateral': bilateral
    }

def find_corners(images, pattern_size):
    """检测图像中的角点，使用多种方法处理光照不均"""
    world_points = create_world_points(pattern_size)
    obj_points = []
    img_points = []
    used_indices = []
    unused_images = []

    # 确保输出目录存在
    processed_dir = './processed_images'
    os.makedirs(processed_dir, exist_ok=True)

    for i, fname in tqdm(enumerate(images), desc="Finding corners", total=len(images)):
        img = cv2.imread(fname)
        if img is None:
            print(f"警告：无法读取图片 {fname}")
            unused_images.append(fname)
            continue
        
        # 预处理图像
        processed_images = preprocess_image(img)
        
        # 尝试多种检测方法
        corners = None
        method_used = None
        
        # 方法1：原始图像
        ret, corners = cv2.findCirclesGrid(processed_images['original'], pattern_size, 
                                          flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret:
            method_used = 'original'
        
        # 方法2：CLAHE处理
        if not ret:
            ret, corners = cv2.findCirclesGrid(processed_images['clahe'], pattern_size, 
                                              flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            if ret:
                method_used = 'clahe'
        
        # 方法3：直方图均衡化
        if not ret:
            ret, corners = cv2.findCirclesGrid(processed_images['equalized'], pattern_size, 
                                              flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            if ret:
                method_used = 'equalized'
        
        # 方法4：双边滤波
        if not ret:
            ret, corners = cv2.findCirclesGrid(processed_images['bilateral'], pattern_size, 
                                              flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            if ret:
                method_used = 'bilateral'
        
        # 方法5：高斯滤波
        if not ret:
            ret, corners = cv2.findCirclesGrid(processed_images['blurred'], pattern_size, 
                                              flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
            if ret:
                method_used = 'blurred'
        
        # 方法6：尝试不同的检测标志
        if not ret:
            ret, corners = cv2.findCirclesGrid(processed_images['clahe'], pattern_size, 
                                              flags=cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING)
            if ret:
                method_used = 'clahe_clustering'
        
        if ret:
            obj_points.append(world_points)
            # 确保corners是(N, 2)格式
            if len(corners.shape) == 3:
                corners = corners.reshape(-1, 2)
            img_points.append(corners)
            used_indices.append(i)
            
            print(f"成功检测到角点: {fname} (使用方法: {method_used})")
            
            # 绘制角点
            cv2.drawChessboardCorners(img, pattern_size, corners.reshape(-1, 1, 2), ret)
            output_fname = os.path.join(processed_dir, f"{i:03d}.png")
            cv2.imwrite(output_fname, img)
        else:
            print(f"无法检测到角点: {fname}")
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
    return world_points * 0.05  # 50mm间距

def sort_images(images):
    """按文件名中的数字排序图像"""
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else 0
    return sorted(images, key=extract_number)

def save_to_excel(obj_points, img_points, used_indices, unused_images, img_size):
    """将处理结果保存到Excel文件"""
    # 创建Excel结果文件夹
    excel_dir = './excel_results'
    os.makedirs(excel_dir, exist_ok=True)
    
    excel_filename = os.path.join(excel_dir, 'image_processing_results.xlsx')
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # 1. 处理摘要表
        summary_data = {
            '项目': ['总图像数量', '成功处理图像数量', '未处理图像数量', '图像宽度', '图像高度', '标定板宽度', '标定板高度', '角点间距(mm)'],
            '数值': [len(used_indices) + len(unused_images), len(obj_points), len(unused_images), 
                    img_size[0], img_size[1], 5, 4, 50]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='处理摘要', index=False)
        
        # 2. 成功处理的图像列表
        if used_indices:
            success_data = {
                '图像索引': used_indices,
                '图像文件名': [f'{idx:03d}.png' for idx in used_indices],
                '角点数量': [len(obj_points[i]) for i in range(len(obj_points))]
            }
            success_df = pd.DataFrame(success_data)
            success_df.to_excel(writer, sheet_name='成功处理图像', index=False)
        
        # 3. 未处理的图像列表
        if unused_images:
            unused_data = {
                '图像文件名': unused_images,
                '状态': ['角点检测失败'] * len(unused_images)
            }
            unused_df = pd.DataFrame(unused_data)
            unused_df.to_excel(writer, sheet_name='未处理图像', index=False)
        
        # 4. 世界坐标点（标定板坐标）
        world_points = create_world_points((5, 4))
        world_data = {
            '点索引': range(len(world_points)),
            'X坐标(mm)': world_points[:, 0],
            'Y坐标(mm)': world_points[:, 1],
            'Z坐标(mm)': world_points[:, 2]
        }
        world_df = pd.DataFrame(world_data)
        world_df.to_excel(writer, sheet_name='世界坐标点', index=False)
        
        # 5. 每张图像的角点坐标详情
        if obj_points and img_points:
            all_corners_data = []
            for img_idx, (obj_pts, img_pts) in enumerate(zip(obj_points, img_points)):
                # img_pts应该已经是(N, 2)格式，但为了安全起见再次检查
                if len(img_pts.shape) == 3:
                    img_pts = img_pts.reshape(-1, 2)
                
                for corner_idx in range(len(obj_pts)):
                    all_corners_data.append({
                        '图像索引': used_indices[img_idx],
                        '角点索引': corner_idx,
                        '世界坐标X(mm)': obj_pts[corner_idx, 0],
                        '世界坐标Y(mm)': obj_pts[corner_idx, 1],
                        '世界坐标Z(mm)': obj_pts[corner_idx, 2],
                        '图像坐标X(像素)': img_pts[corner_idx, 0],
                        '图像坐标Y(像素)': img_pts[corner_idx, 1]
                    })
            
            corners_df = pd.DataFrame(all_corners_data)
            corners_df.to_excel(writer, sheet_name='角点坐标详情', index=False)
    
    print(f"Excel文件已保存: {excel_filename}")
    return excel_filename

def save_processing_results(obj_points, img_points, used_indices, unused_images, img_size):
    """保存图像处理结果"""
    # 创建数据文件夹
    data_dir = './processing_data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 保存角点数据
    np.save(os.path.join(data_dir, 'obj_points.npy'), obj_points)
    np.save(os.path.join(data_dir, 'img_points.npy'), img_points)
    np.save(os.path.join(data_dir, 'used_indices.npy'), used_indices)
    np.save(os.path.join(data_dir, 'img_size.npy'), img_size)
    
    # 保存到Excel
    excel_filename = save_to_excel(obj_points, img_points, used_indices, unused_images, img_size)
    
    # 保存处理统计信息
    with open('./processing_summary.txt', 'w', encoding='utf-8') as f:
        f.write("图像处理结果摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"总图像数量: {len(used_indices) + len(unused_images)}\n")
        f.write(f"成功处理图像数量: {len(obj_points)}\n")
        f.write(f"未处理图像数量: {len(unused_images)}\n")
        f.write(f"图像尺寸: {img_size[0]} x {img_size[1]}\n")
        f.write(f"标定板尺寸: 5 x 4\n")
        f.write(f"角点间距: 50mm\n")
        f.write(f"Excel文件: {excel_filename}\n")
        f.write("\n")
        
        if unused_images:
            f.write("未处理的图像:\n")
            for img in unused_images:
                f.write(f"  - {img}\n")
        
        f.write("\n成功处理的图像索引:\n")
        for idx in used_indices:
            f.write(f"  - {idx:03d}.png\n")
    
    print(f"图像处理结果已保存:")
    print(f"  - {data_dir}/obj_points.npy: 世界坐标点")
    print(f"  - {data_dir}/img_points.npy: 图像坐标点")
    print(f"  - {data_dir}/used_indices.npy: 使用的图像索引")
    print(f"  - {data_dir}/img_size.npy: 图像尺寸")
    print(f"  - processing_summary.txt: 处理摘要")
    print(f"  - {excel_filename}: Excel详细结果")
    print(f"  - ./processed_images/: 处理后的图像（带角点标记）")

def main():
    """主函数：图像处理和角点检测"""
    pattern_size = (5, 4)  # 5x4标定板
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
