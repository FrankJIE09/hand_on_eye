import cv2
import numpy as np
import yaml
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def load_robot_poses(used_indices):
    """加载机器人位姿数据"""
    robot_rot_matrices = []
    robot_trans_vectors = []
    
    for i in tqdm(used_indices, desc="Loading robot poses"):
        pose_file = f'./captured_images/pose_{i}.npy'
        if os.path.exists(pose_file):
            pose_data = np.load(pose_file)
            # 取前6个元素（位置和姿态）
            pose = pose_data[6:12]
            if len(pose) == 6:
                trans_vector = np.array(pose[:3])/1000  # 转换为米
                rot_vector = np.array(pose[3:])
                rotation_matrix = R.from_euler('xyz', rot_vector, degrees=True).as_matrix()
                robot_rot_matrices.append(rotation_matrix)
                robot_trans_vectors.append(trans_vector)
            else:
                print(f"警告：pose_{i}.npy 数据格式不正确，跳过")
        else:
            print(f"警告：找不到 pose_{i}.npy 文件，跳过")
    
    return robot_rot_matrices, robot_trans_vectors

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

def validate_and_fix_rotation_matrix(rotation_matrix):
    """验证并修正旋转矩阵"""
    # 计算行列式
    det = np.linalg.det(rotation_matrix)
    print(f"旋转矩阵行列式: {det:.6f}")
    
    # 如果行列式为负值（左手坐标系），需要修正
    if det < 0:
        print("警告：检测到左手坐标系，正在修正...")
        # 对矩阵进行SVD分解
        U, s, Vt = np.linalg.svd(rotation_matrix)
        # 强制行列式为正值（右手坐标系）
        if np.linalg.det(U @ Vt) < 0:
            # 翻转最后一列
            U[:, -1] *= -1
        rotation_matrix = U @ Vt
        print(f"修正后旋转矩阵行列式: {np.linalg.det(rotation_matrix):.6f}")
    
    # 验证正交性
    should_be_identity = rotation_matrix @ rotation_matrix.T
    orthogonality_error = np.linalg.norm(should_be_identity - np.eye(3))
    print(f"正交性误差: {orthogonality_error:.6f}")
    
    if orthogonality_error > 1e-6:
        print("警告：旋转矩阵正交性较差，使用SVD重新正交化...")
        U, s, Vt = np.linalg.svd(rotation_matrix)
        rotation_matrix = U @ Vt
        # 确保行列式为正
        if np.linalg.det(rotation_matrix) < 0:
            U[:, -1] *= -1
            rotation_matrix = U @ Vt
        print(f"重新正交化后行列式: {np.linalg.det(rotation_matrix):.6f}")
    
    return rotation_matrix

def matrix_to_rpy_manual(rotation_matrix):
    """手动计算旋转矩阵的RPY角度（度）"""
    r11, r12, r13 = rotation_matrix[0, :]
    r21, r22, r23 = rotation_matrix[1, :]
    r31, r32, r33 = rotation_matrix[2, :]
    
    # 计算RPY角度（单位：弧度）
    roll = np.arctan2(r32, r33)
    pitch = np.arctan2(-r31, np.sqrt(r32**2 + r33**2))
    yaw = np.arctan2(r21, r11)
    
    # 转换为度
    return np.array([np.degrees(roll), np.degrees(pitch), np.degrees(yaw)])

def hand_eye_calibration(robot_rot_matrices, robot_trans_vectors, cam_rot_matrices, cam_trans_vectors,
                         method=cv2.CALIB_HAND_EYE_TSAI):
    """手眼标定"""
    
    # 将旋转矩阵和平移向量组合成SE(3)矩阵，然后求逆
    print("将位姿数据组合成SE(3)矩阵并求逆...")
    
    # 机器人位姿SE(3)矩阵列表
    robot_se3_matrices = []
    for i in range(len(robot_rot_matrices)):
        se3_matrix = rot_trans_to_se3(robot_rot_matrices[i], robot_trans_vectors[i])
        inv_se3_matrix = np.linalg.inv(se3_matrix)
        robot_se3_matrices.append(inv_se3_matrix)
    
    # 相机位姿SE(3)矩阵列表
    cam_se3_matrices = []
    for i in range(len(cam_rot_matrices)):
        se3_matrix = rot_trans_to_se3(cam_rot_matrices[i], cam_trans_vectors[i])
        inv_se3_matrix = np.linalg.inv(se3_matrix)
        cam_se3_matrices.append(inv_se3_matrix)
    
    # 从SE(3)矩阵中提取旋转矩阵和平移向量
    inv_robot_rot_matrices = []
    inv_robot_trans_vectors = []
    inv_cam_rot_matrices = []
    inv_cam_trans_vectors = []
    
    for se3_matrix in robot_se3_matrices:
        rot, trans = se3_to_rot_trans(se3_matrix)
        inv_robot_rot_matrices.append(rot)
        inv_robot_trans_vectors.append(trans)
    
    for se3_matrix in cam_se3_matrices:
        rot, trans = se3_to_rot_trans(se3_matrix)
        inv_cam_rot_matrices.append(rot)
        inv_cam_trans_vectors.append(trans)

    rm, tm = cv2.calibrateHandEye(inv_robot_rot_matrices, inv_robot_trans_vectors, cam_rot_matrices,
                                  cam_trans_vectors, method=method)
    
    # 验证并修正旋转矩阵
    print("验证手眼标定旋转矩阵...")
    rm = validate_and_fix_rotation_matrix(rm)
    
    transform_matrix = create_transformation_matrix(rm, tm)
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    
    # 验证并修正逆变换矩阵的旋转部分
    print("验证逆变换矩阵旋转部分...")
    inv_rm = validate_and_fix_rotation_matrix(inv_transform_matrix[:3, :3])
    inv_transform_matrix[:3, :3] = inv_rm
    
    # 现在安全地计算RPY
    try:
        rpy = R.from_matrix(rm).as_euler('xyz', degrees=True)
        inv_rpy = R.from_matrix(inv_rm).as_euler('xyz', degrees=True)
    except ValueError as e:
        print(f"旋转矩阵转换为欧拉角时出错: {e}")
        # 作为备选方案，使用atan2计算RPY
        rpy = matrix_to_rpy_manual(rm)
        inv_rpy = matrix_to_rpy_manual(inv_rm)
        print("使用手动计算的RPY角度")
    
    # 计算手眼标定误差（使用原始位姿数据）
    hand_eye_error = calculate_hand_eye_error(robot_rot_matrices, robot_trans_vectors, 
                                            cam_rot_matrices, cam_trans_vectors, rm, tm)
    
    return transform_matrix, inv_transform_matrix, rpy, inv_rpy, hand_eye_error

def calculate_hand_eye_error(robot_rot_matrices, robot_trans_vectors, cam_rot_matrices, cam_trans_vectors, rm, tm):
    """计算手眼标定的平均误差"""
    total_error = 0
    count = 0
    
    for i in range(len(robot_rot_matrices)):
        # 计算理论值
        theoretical_rot = robot_rot_matrices[i] @ rm
        theoretical_trans = robot_rot_matrices[i] @ tm + robot_trans_vectors[i]
        
        # 计算实际值
        actual_rot = rm @ cam_rot_matrices[i]
        actual_trans = rm @ cam_trans_vectors[i] + tm
        
        # 旋转误差（角度差）
        try:
            theoretical_rpy = R.from_matrix(theoretical_rot).as_euler('xyz', degrees=True)
            actual_rpy = R.from_matrix(actual_rot).as_euler('xyz', degrees=True)
        except ValueError:
            # 如果scipy方法失败，使用手动计算
            theoretical_rpy = matrix_to_rpy_manual(theoretical_rot)
            actual_rpy = matrix_to_rpy_manual(actual_rot)
        rot_error = np.linalg.norm(theoretical_rpy - actual_rpy)
        
        # 平移误差（毫米）
        trans_error = np.linalg.norm(theoretical_trans - actual_trans) * 1000
        
        total_error += rot_error + trans_error
        count += 1
    
    return total_error / count if count > 0 else 0

def create_transformation_matrix(rotation_matrix, translation_vector):
    """创建4x4变换矩阵"""
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translation_vector.reshape(-1)
    return transformation_matrix

def se3_to_rot_trans(se3_matrix):
    """从SE(3)矩阵中提取旋转矩阵和平移向量"""
    rotation_matrix = se3_matrix[:3, :3]
    translation_vector = se3_matrix[:3, 3]
    return rotation_matrix, translation_vector

def rot_trans_to_se3(rotation_matrix, translation_vector):
    """将旋转矩阵和平移向量组合成SE(3)矩阵"""
    se3_matrix = np.eye(4)
    se3_matrix[:3, :3] = rotation_matrix
    se3_matrix[:3, 3] = translation_vector.reshape(-1)
    return se3_matrix

def save_calibration_to_yaml_and_txt(yaml_filename, txt_filename, intrinsic_matrix, distortion_coeffs,
                                     transform_matrix, inv_transform_matrix, rpy, inv_rpy, 
                                     reprojection_error, hand_eye_error):
    """保存标定结果到YAML和TXT文件"""
    calibration_data = {
        'camera_matrix': intrinsic_matrix.tolist(),
        'distortion_coefficients': distortion_coeffs.tolist(),
        'hand_eye_transformation_matrix': transform_matrix.tolist(),
        'hand_eye_rpy': rpy.tolist(),
        'inverse_hand_eye_transformation_matrix': inv_transform_matrix.tolist(),
        'inverse_hand_eye_rpy': inv_rpy.tolist(),
        'calibration_accuracy': {
            'camera_reprojection_error_pixels': float(reprojection_error),
            'hand_eye_calibration_error': float(hand_eye_error),
            'hand_eye_rotation_error_degrees': float(hand_eye_error / 2),
            'hand_eye_translation_error_mm': float(hand_eye_error / 2)
        }
    }
    
    with open(yaml_filename, 'w', encoding='utf-8') as f:
        yaml.dump(calibration_data, f, default_flow_style=None, allow_unicode=True)
    print(f"标定结果已保存到 {yaml_filename}")
    
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("Camera Matrix (Intrinsic):\n")
        np.savetxt(f, intrinsic_matrix, fmt='%f')
        f.write("\nDistortion Coefficients:\n")
        np.savetxt(f, distortion_coeffs, fmt='%f')
        f.write("\nHand-Eye Transformation Matrix:\n")
        np.savetxt(f, transform_matrix, fmt='%f')
        f.write("\nHand-Eye RPY:\n")
        np.savetxt(f, rpy, fmt='%f')
        f.write("\nInverse Hand-Eye Transformation Matrix:\n")
        np.savetxt(f, inv_transform_matrix, fmt='%f')
        f.write("\nInverse Hand-Eye RPY:\n")
        np.savetxt(f, inv_rpy, fmt='%f')
        f.write(f"\nCalibration Accuracy:\n")
        f.write(f"Camera Reprojection Error: {reprojection_error:.6f} pixels\n")
        f.write(f"Hand-Eye Calibration Error: {hand_eye_error:.6f}\n")
    print(f"标定结果已保存到 {txt_filename}")

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
    """主函数：标定计算"""
    print("开始标定计算...")
    
    # 加载图像处理结果
    obj_points, img_points, used_indices, img_size = load_processing_results()
    if obj_points is None:
        return
    
    try:
        # 加载机器人位姿数据
        robot_rot_matrices, robot_trans_vectors = load_robot_poses(used_indices)
        if len(robot_rot_matrices) == 0:
            print("错误：没有有效的机器人位姿数据")
            return
        
        print(f"成功加载 {len(robot_rot_matrices)} 个机器人位姿")
        
        # 相机标定
        print("进行相机标定...")
        ret, intrinsic_matrix, distortion_coeffs, optimal_matrix, trans_vectors, cam_rot_matrices, reprojection_error = calibrate_camera(obj_points, img_points, img_size)
        
        # 手眼标定
        print("进行手眼标定...")
        transform_matrix, inv_transform_matrix, rpy, inv_rpy, hand_eye_error = hand_eye_calibration(robot_rot_matrices, robot_trans_vectors, cam_rot_matrices, trans_vectors)
        
        # 显示结果
        print("\n标定结果:")
        print(f"相机标定重投影误差: {reprojection_error:.6f} 像素")
        print(f"手眼标定误差: {hand_eye_error:.6f}")
        print(f"手眼变换矩阵:")
        print(transform_matrix)
        print(f"手眼RPY (度): {rpy}")
        
        # 保存结果
        save_calibration_to_yaml_and_txt('./config.yaml', 'calibration_results.txt', 
                                        intrinsic_matrix, distortion_coeffs, 
                                        transform_matrix, inv_transform_matrix, rpy, inv_rpy, 
                                        reprojection_error, hand_eye_error)
        
        print("\n标定完成！")
                                        
    except Exception as e:
        print(f"标定过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
