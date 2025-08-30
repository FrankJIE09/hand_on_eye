import cv2
import numpy as np
import glob
import os
from scipy.spatial.transform import Rotation as R
import re
import yaml
from tqdm import tqdm

def find_corners(images, pattern_size):
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
            # 修复：正确绘制角点
            cv2.drawChessboardCorners(img, pattern_size, corners.reshape(-1, 1, 2), ret)
            output_fname = f"./processed_images/{i:03d}.png"
            cv2.imwrite(output_fname, img)
        else:
            unused_images.append(fname)

    return obj_points, img_points, used_indices, unused_images

def create_world_points(pattern_size):
    width, height = pattern_size
    world_points = np.zeros((width * height, 3), np.float32)
    num = 0
    for i in range(height):
        for j in range(width):
            world_points[num, :2] = [j + 0.5 * (i % 2), i * 0.5]
            num += 1
    return world_points * 0.02

def load_robot_poses(file_path, used_indices):
    data = np.load(file_path, allow_pickle=True)
    robot_rot_matrices = []
    robot_trans_vectors = []
    valid_indices = [i for i in used_indices if i < len(data)]
    for i in tqdm(valid_indices, desc="Loading robot poses"):
        pose = data[i]
        if len(pose) == 6:
            trans_vector = np.array(pose[:3])/1000
            rot_vector = np.array(pose[3:])
            rotation_matrix = R.from_euler('xyz', rot_vector, degrees=True).as_matrix()
            robot_rot_matrices.append(rotation_matrix)
            robot_trans_vectors.append(trans_vector)
        else:
            raise ValueError("每个pose数据应包含6个元素（3个平移+3个旋转）")
    return robot_rot_matrices, robot_trans_vectors

def calibrate_camera(obj_points, img_points, img_size):
    ret, intrinsic_matrix, distortion_coeffs, rot_vectors, trans_vectors = cv2.calibrateCamera(obj_points, img_points,
                                                                                               img_size, None, None)
    if not ret:
        raise RuntimeError("相机标定失败")
    
    optimal_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, img_size, 0, img_size)
    cam_rot_matrices = [cv2.Rodrigues(rot_vec)[0] for rot_vec in rot_vectors]
    
    # 计算重投影误差
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rot_vectors[i], trans_vectors[i], intrinsic_matrix, distortion_coeffs)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    reprojection_error = mean_error / len(obj_points)
    
    return ret, intrinsic_matrix, distortion_coeffs, optimal_matrix, trans_vectors, cam_rot_matrices, reprojection_error

def hand_eye_calibration(robot_rot_matrices, robot_trans_vectors, cam_rot_matrices, cam_trans_vectors,
                         method=cv2.CALIB_HAND_EYE_PARK):
    rm, tm = cv2.calibrateHandEye(robot_rot_matrices, robot_trans_vectors, cam_rot_matrices,
                                  cam_trans_vectors, method=method)
    transform_matrix = create_transformation_matrix(rm, tm)
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    rpy = R.from_matrix(rm).as_euler('xyz', degrees=True)
    inv_rpy = R.from_matrix(inv_transform_matrix[:3, :3]).as_euler('xyz', degrees=True)
    
    # 计算手眼标定误差
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
        rot_error = np.linalg.norm(R.from_matrix(theoretical_rot).as_euler('xyz', degrees=True) - 
                                  R.from_matrix(actual_rot).as_euler('xyz', degrees=True))
        
        # 平移误差（毫米）
        trans_error = np.linalg.norm(theoretical_trans - actual_trans) * 1000
        
        total_error += rot_error + trans_error
        count += 1
    
    return total_error / count if count > 0 else 0

def create_transformation_matrix(rotation_matrix, translation_vector):
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translation_vector.reshape(-1)
    return transformation_matrix

def save_calibration_to_yaml_and_txt(yaml_filename, txt_filename, intrinsic_matrix, distortion_coeffs,
                                     transform_matrix, inv_transform_matrix, rpy, inv_rpy, 
                                     reprojection_error, hand_eye_error):
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
            'hand_eye_rotation_error_degrees': float(hand_eye_error / 2),  # 假设旋转和平移误差各占一半
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

def sort_images(images):
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else 0
    return sorted(images, key=extract_number)

def main():
    pattern_size = (4, 11)
    images = glob.glob('./captured_images/*.png')
    
    if not images:
        print("错误：在 ./captured_images/ 目录中未找到图片文件")
        return
        
    images = sort_images(images)
    print(f"找到 {len(images)} 张图片")
    
    obj_points, img_points, used_indices, unused_images = find_corners(images, pattern_size)
    
    if len(obj_points) == 0:
        print("错误：没有找到有效的角点，无法进行标定")
        return
        
    if unused_images:
        print(f"未处理的图片数量: {len(unused_images)}")
        print(f"未处理的图片: {unused_images}")
    
    print(f"成功处理 {len(obj_points)} 张图片")
    
    try:
        robot_rot_matrices, robot_trans_vectors = load_robot_poses('./pose_data.npy', used_indices)
        if len(robot_rot_matrices) == 0:
            print("错误：没有有效的机器人位姿数据")
            return
            
        img_size = cv2.imread(images[0]).shape[::-1][1:3]
        ret, intrinsic_matrix, distortion_coeffs, optimal_matrix, trans_vectors, cam_rot_matrices, reprojection_error = calibrate_camera(obj_points, img_points, img_size)
        
        transform_matrix, inv_transform_matrix, rpy, inv_rpy, hand_eye_error = hand_eye_calibration(robot_rot_matrices, robot_trans_vectors, cam_rot_matrices, trans_vectors)
        
        print(f"相机标定重投影误差: {reprojection_error:.6f} 像素")
        print(f"手眼标定误差: {hand_eye_error:.6f}")
        
        save_calibration_to_yaml_and_txt('./config.yaml', 'calibration_results.txt', intrinsic_matrix, distortion_coeffs, 
                                        transform_matrix, inv_transform_matrix, rpy, inv_rpy, 
                                        reprojection_error, hand_eye_error)
                                        
    except Exception as e:
        print(f"标定过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
