import cv2
import numpy as np
import yaml
import os


def calibrate_camera(obj_points, img_points, img_size):
    """相机标定"""
    ret, intrinsic_matrix, distortion_coeffs, rot_vectors, trans_vectors = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)
    
    if not ret:
        raise RuntimeError("相机标定失败")
    
    optimal_matrix, roi = cv2.getOptimalNewCameraMatrix(
        intrinsic_matrix, distortion_coeffs, img_size, 0, img_size)
    cam_rot_matrices = [cv2.Rodrigues(rot_vec)[0] for rot_vec in rot_vectors]
    
    # 计算重投影误差
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(
            obj_points[i], rot_vectors[i], trans_vectors[i], 
            intrinsic_matrix, distortion_coeffs)
        # 确保数据类型和形状一致
        img_points_i = np.array(img_points[i], dtype=np.float32)
        imgpoints2 = np.array(imgpoints2, dtype=np.float32).reshape(-1, 2)
        error = cv2.norm(img_points_i, imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    reprojection_error = mean_error / len(obj_points)
    
    return ret, intrinsic_matrix, distortion_coeffs, optimal_matrix, trans_vectors, cam_rot_matrices, reprojection_error


def save_calibration_to_yaml_and_txt(yaml_filename, txt_filename, intrinsic_matrix, distortion_coeffs, reprojection_error):
    """保存相机内参标定结果到YAML和TXT文件"""
    calibration_data = {
        'camera_matrix': intrinsic_matrix.tolist(),
        'distortion_coefficients': distortion_coeffs.tolist(),
        'calibration_accuracy': {
            'camera_reprojection_error_pixels': float(reprojection_error)
        }
    }
    
    with open(yaml_filename, 'w', encoding='utf-8') as f:
        yaml.dump(calibration_data, f, default_flow_style=None, allow_unicode=True)
    print(f"相机内参标定结果已保存到 {yaml_filename}")
    
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("=== 梅卡相机内参标定结果 ===\n\n")
        f.write("Camera Matrix (Intrinsic):\n")
        np.savetxt(f, intrinsic_matrix, fmt='%f')
        f.write("\nDistortion Coefficients:\n")
        np.savetxt(f, distortion_coeffs, fmt='%f')
        f.write(f"\nCalibration Accuracy:\n")
        f.write(f"Camera Reprojection Error: {reprojection_error:.6f} pixels\n")
        f.write(f"\n标定板规格: 5x4, 50mm间距\n")
        f.write(f"标定时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"相机内参标定结果已保存到 {txt_filename}")

def load_processing_results():
    """加载图像处理结果"""
    try:
        # 从processing_data文件夹加载数据
        data_dir = './processing_data'
        obj_points = np.load(os.path.join(data_dir, 'obj_points.npy'), allow_pickle=True)
        img_points = np.load(os.path.join(data_dir, 'img_points.npy'), allow_pickle=True)
        used_indices = np.load(os.path.join(data_dir, 'used_indices.npy'), allow_pickle=True)
        img_size = np.load(os.path.join(data_dir, 'img_size.npy'), allow_pickle=True)
        
        print(f"成功加载图像处理结果:")
        print(f"  - 角点数量: {len(obj_points)}")
        print(f"  - 图像尺寸: {img_size[0]} x {img_size[1]}")
        print(f"  - 使用的图像索引: {len(used_indices)}")
        
        return obj_points, img_points, used_indices, img_size
        
    except FileNotFoundError as e:
        print(f"错误：找不到图像处理结果文件 {e}")
        print("请先运行 step3_image_processing.py 进行图像处理")
        return None, None, None, None

def main():
    """主函数：相机内参标定"""
    print("开始梅卡相机内参标定...")
    
    # 加载图像处理结果
    obj_points, img_points, used_indices, img_size = load_processing_results()
    if obj_points is None:
        return
    
    try:
        # 相机内参标定
        print("进行相机内参标定...")
        ret, intrinsic_matrix, distortion_coeffs, optimal_matrix, trans_vectors, cam_rot_matrices, reprojection_error = calibrate_camera(obj_points, img_points, img_size)
        
        # 显示结果
        print("\n相机内参标定结果:")
        print(f"相机内参矩阵:")
        print(intrinsic_matrix)
        print(f"畸变系数:")
        print(distortion_coeffs)
        print(f"重投影误差: {reprojection_error:.6f} 像素")
        
        # 保存结果
        save_calibration_to_yaml_and_txt('./camera_intrinsics.yaml', 'camera_intrinsics_results.txt', 
                                        intrinsic_matrix, distortion_coeffs, reprojection_error)
        
        print("\n相机内参标定完成！")
        print("结果已保存到:")
        print("  - camera_intrinsics.yaml: YAML格式的内参文件")
        print("  - camera_intrinsics_results.txt: 文本格式的内参文件")
                                        
    except Exception as e:
        print(f"标定过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
