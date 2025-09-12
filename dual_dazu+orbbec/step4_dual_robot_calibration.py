import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import yaml
from tqdm import tqdm


def load_config(config_file="config.yaml"):
    """
    加载配置文件
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_camera_calibration_results(file_path="camera_calibration_results.npz"):
    """
    加载相机标定结果
    """
    try:
        data = np.load(file_path, allow_pickle=True)
        intrinsic_matrix = data['intrinsic_matrix']
        distortion_coeffs = data['distortion_coeffs']
        trans_vectors = data['trans_vectors']
        cam_rot_matrices = data['cam_rot_matrices']
        reprojection_error = float(data['reprojection_error'])
        used_indices = data['used_indices']
        
        print(f"成功加载相机标定结果")
        print(f"重投影误差: {reprojection_error:.6f} 像素")
        print(f"有效图片数量: {len(used_indices)}")
        
        return intrinsic_matrix, distortion_coeffs, trans_vectors, cam_rot_matrices, reprojection_error, used_indices
    except Exception as e:
        print(f"加载相机标定结果失败: {e}")
        return None, None, None, None, None, None


def load_hand_eye_calibration_results(file_path="calibration_results_robot1_to_camera.txt"):
    """
    加载手眼标定结果
    """
    hand_eye_transform_matrix = np.eye(4)
    hand_eye_rpy = np.array([0, 0, 0])

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 解析手眼变换矩阵
        matrix_start = -1
        for i, line in enumerate(lines):
            if "Hand-Eye Transformation Matrix:" in line:
                matrix_start = i + 1
                break

        if matrix_start != -1:
            matrix_lines = lines[matrix_start:matrix_start + 4]
            for i, line in enumerate(matrix_lines):
                values = [float(x) for x in line.strip().split()]
                hand_eye_transform_matrix[i, :] = values

        # 解析RPY角度
        rpy_start = -1
        for i, line in enumerate(lines):
            if "Hand-Eye RPY:" in line:
                rpy_start = i + 1
                break

        if rpy_start != -1:
            rpy_values = []
            for i in range(3):
                rpy_values.append(float(lines[rpy_start + i].strip()))
            hand_eye_rpy = np.array(rpy_values)

        print(f"成功加载手眼标定结果")
        print(f"手眼变换矩阵:")
        print(hand_eye_transform_matrix)

    except Exception as e:
        print(f"加载手眼标定结果失败: {e}")
        print("使用默认值")

    return hand_eye_transform_matrix, hand_eye_rpy


def load_dual_robot_poses_from_images_dir(images_dir, used_indices):
    """
    从captured_images目录中加载双机械臂位姿数据
    """
    robot1_rot_matrices = []
    robot1_trans_vectors = []
    robot2_rot_matrices = []
    robot2_trans_vectors = []

    for i in tqdm(used_indices, desc="Loading dual robot poses from images directory"):
        robot1_pose_file = f"{images_dir}robot1_pose_{i:03d}.npy"
        robot2_pose_file = f"{images_dir}robot2_pose_{i:03d}.npy"

        # 检查文件是否存在
        if not os.path.exists(robot1_pose_file) or not os.path.exists(robot2_pose_file):
            print(f"警告：位置 {i} 的位姿文件不存在，跳过")
            continue

        try:
            # 加载位姿数据
            pose1 = np.load(robot1_pose_file)[6:12]
            pose2 = np.load(robot2_pose_file)[6:12]

            # 确保位姿数据是6DOF格式
            if len(pose1) >= 6 and len(pose2) >= 6:
                # 取前6个元素（x, y, z, rx, ry, rz）
                pose1_6dof = pose1[:6]
                pose2_6dof = pose2[:6]

                # 机械臂1位姿
                trans_vector1 = np.array(pose1_6dof[:3]) / 1000  # 转换为米
                rot_vector1 = np.array(pose1_6dof[3:])
                rotation_matrix1 = R.from_euler('xyz', rot_vector1, degrees=True).as_matrix()
                robot1_rot_matrices.append(rotation_matrix1)
                robot1_trans_vectors.append(trans_vector1)

                # 机械臂2位姿
                trans_vector2 = np.array(pose2_6dof[:3]) / 1000  # 转换为米
                rot_vector2 = np.array(pose2_6dof[3:])
                rotation_matrix2 = R.from_euler('xyz', rot_vector2, degrees=True).as_matrix()
                robot2_rot_matrices.append(rotation_matrix2)
                robot2_trans_vectors.append(trans_vector2)
            else:
                print(f"警告：位置 {i} 的位姿数据格式不正确，跳过")
                continue

        except Exception as e:
            print(f"警告：加载位置 {i} 的位姿数据失败: {e}，跳过")
            continue

    print(f"成功加载 {len(robot1_rot_matrices)} 个有效的位姿对")
    return (robot1_rot_matrices, robot1_trans_vectors,
            robot2_rot_matrices, robot2_trans_vectors)


def create_transformation_matrix(rotation_matrix, translation_vector):
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translation_vector.reshape(-1)
    return transformation_matrix


def dual_robot_base_calibration(robot1_rot_matrices, robot1_trans_vectors,
                                robot2_rot_matrices, robot2_trans_vectors,
                                intrinsic_matrix, distortion_coeffs,
                                trans_vectors, cam_rot_matrices,
                                hand_eye_transform_matrix, hand_eye_rpy,
                                method=cv2.CALIB_HAND_EYE_PARK):
    """
    双机械臂base坐标系标定
    计算从机械臂2 base到机械臂1 base的变换矩阵
    """

    # 将SO(3)旋转矩阵和R³平移向量组合成SE(3)变换矩阵

    # 1. 机械臂1位姿变换矩阵列表 (Base_robot1 → Flange_robot1)
    robot1_poses = []
    for i in range(len(robot1_rot_matrices)):
        T_robot1 = create_transformation_matrix(robot1_rot_matrices[i], robot1_trans_vectors[i])
        robot1_poses.append(T_robot1)

    # 2. 机械臂2位姿变换矩阵列表 (Base_robot2 → Flange_robot2)
    robot2_poses = []
    for i in range(len(robot2_rot_matrices)):
        T_robot2 = create_transformation_matrix(robot2_rot_matrices[i], robot2_trans_vectors[i])
        robot2_poses.append(T_robot2)

    # 3. 相机到标定板变换矩阵列表 (Camera → Calibration_board)
    camera_to_board_poses = []
    for i in range(len(cam_rot_matrices)):
        T_camera_to_board = create_transformation_matrix(cam_rot_matrices[i], trans_vectors[i])
        camera_to_board_poses.append(T_camera_to_board)

    # 4. 手眼标定变换矩阵 (Flange_robot1 → Camera)
    T_flange1_to_camera = hand_eye_transform_matrix

    # 将列表转换为numpy数组
    camera_to_board_poses_array = np.array(camera_to_board_poses)
    robot1_poses_array = np.array(robot1_poses)
    robot2_poses_array = np.array(robot2_poses)

    # 计算变换链
    # w1 = np.linalg.inv(camera_to_board_poses_array) @ np.linalg.inv(T_flange1_to_camera) @ np.linalg.inv(
    #     robot1_poses_array)
    # w1 = robot1_poses_array @ T_flange1_to_camera @ camera_to_board_poses_array
    w1 = np.linalg.inv(robot2_poses_array)
    w2 = robot1_poses_array@T_flange1_to_camera@camera_to_board_poses_array
    # w2 =robot2_poses_array

    # 提取旋转矩阵和平移向量
    w1_rot_matrices = w1[:, :3, :3]  # 旋转矩阵部分
    w1_trans_vectors = w1[:, :3, 3]  # 平移向量部分
    w2_rot_matrices = w2[:, :3, :3]  # 旋转矩阵部分
    w2_trans_vectors = w2[:, :3, 3]  # 平移向量部分

    # 手眼标定
    rm, tm = cv2.calibrateHandEye(w1_rot_matrices, w1_trans_vectors,
                                  w2_rot_matrices, w2_trans_vectors,
                                  method=method)

    print("=== 平移向量 tm ===")
    print("tm =", tm)
    print("=== 四元数表示 ===")
    print("quaternion =", R.from_matrix(rm).as_quat())
    print("=== 轴角表示 ===")
    print("axis_angle =", R.from_matrix(rm).as_rotvec(degrees=True))

    # 构建变换矩阵
    transform_matrix = create_transformation_matrix(rm, tm)
    inv_transform_matrix = np.linalg.inv(transform_matrix)

    # 计算RPY角度
    rpy = R.from_matrix(rm).as_euler('xyz', degrees=True)
    inv_rpy = R.from_matrix(inv_transform_matrix[:3, :3]).as_euler('xyz', degrees=True)

    # 计算标定误差
    calibration_error = calculate_dual_robot_error(w1_rot_matrices, w1_trans_vectors,
                                                   w2_rot_matrices, w2_trans_vectors, rm, tm)

    return transform_matrix, inv_transform_matrix, rpy, inv_rpy, calibration_error


def calculate_dual_robot_error(robot1_rot_matrices, robot1_trans_vectors,
                               robot2_rot_matrices, robot2_trans_vectors, rm, tm):
    """计算双机械臂标定的平均误差"""
    total_error = 0
    count = 0

    for i in range(len(robot1_rot_matrices)):
        # 计算理论值：robot1_pose = T_robot2_to_robot1 * robot2_pose
        theoretical_rot = rm @ robot2_rot_matrices[i]
        theoretical_trans = rm @ robot2_trans_vectors[i] + tm

        # 计算实际值
        actual_rot = robot1_rot_matrices[i]
        actual_trans = robot1_trans_vectors[i]

        # 旋转误差（角度差）
        rot_error = np.linalg.norm(R.from_matrix(theoretical_rot).as_euler('xyz', degrees=True) -
                                   R.from_matrix(actual_rot).as_euler('xyz', degrees=True))

        # 平移误差（毫米）
        trans_error = np.linalg.norm(theoretical_trans - actual_trans) * 1000

        total_error += rot_error + trans_error
        count += 1

    return total_error / count if count > 0 else 0


def save_dual_robot_calibration_to_txt(txt_filename, intrinsic_matrix, distortion_coeffs,
                                       transform_matrix, inv_transform_matrix, rpy, inv_rpy,
                                       reprojection_error, dual_robot_error):
    """
    保存双机械臂标定结果到文本文件
    """
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("Dual Robot Base Calibration Results\n")
        f.write("===================================\n\n")

        f.write("Camera Matrix (Intrinsic):\n")
        np.savetxt(f, intrinsic_matrix, fmt='%f')
        f.write("\nDistortion Coefficients:\n")
        np.savetxt(f, distortion_coeffs, fmt='%f')

        f.write("\nRobot2 Base to Robot1 Base Transformation Matrix:\n")
        np.savetxt(f, transform_matrix, fmt='%f')
        f.write("\nRobot2 Base to Robot1 Base RPY (degrees):\n")
        np.savetxt(f, rpy, fmt='%f')

        f.write("\nRobot1 Base to Robot2 Base Transformation Matrix:\n")
        np.savetxt(f, inv_transform_matrix, fmt='%f')
        f.write("\nRobot1 Base to Robot2 Base RPY (degrees):\n")
        np.savetxt(f, inv_rpy, fmt='%f')

        f.write(f"\nCalibration Accuracy:\n")
        f.write(f"Camera Reprojection Error: {reprojection_error:.6f} pixels\n")
        f.write(f"Dual Robot Calibration Error: {dual_robot_error:.6f}\n")
        f.write(f"Dual Robot Rotation Error: {dual_robot_error / 2:.6f} degrees\n")
        f.write(f"Dual Robot Translation Error: {dual_robot_error / 2:.6f} mm\n")
    print(f"双机械臂标定结果已保存到 {txt_filename}")


def main():
    # 加载配置
    config_data = load_config()

    # 加载相机标定结果
    intrinsic_matrix, distortion_coeffs, trans_vectors, cam_rot_matrices, reprojection_error, used_indices = load_camera_calibration_results()

    if intrinsic_matrix is None:
        print("错误：无法加载相机标定结果，请先运行 step3_camera_calibration.py")
        return

    # 加载手眼标定结果
    hand_eye_transform_matrix, hand_eye_rpy = load_hand_eye_calibration_results()

    # 从captured_images目录中加载双机械臂位姿数据
    robot1_rot_matrices, robot1_trans_vectors, robot2_rot_matrices, robot2_trans_vectors = load_dual_robot_poses_from_images_dir(
        config_data['paths']['images_dir'],
        used_indices
    )

    if len(robot1_rot_matrices) == 0 or len(robot2_rot_matrices) == 0:
        print("错误：没有有效的双机械臂位姿数据")
        return

    try:
        # 双机械臂base标定
        transform_matrix, inv_transform_matrix, rpy, inv_rpy, dual_robot_error = dual_robot_base_calibration(
            robot1_rot_matrices, robot1_trans_vectors,
            robot2_rot_matrices, robot2_trans_vectors,
            intrinsic_matrix, distortion_coeffs,
            trans_vectors, cam_rot_matrices,
            hand_eye_transform_matrix, hand_eye_rpy
        )

        print(f"相机标定重投影误差: {reprojection_error:.6f} 像素")
        print(f"双机械臂标定误差: {dual_robot_error:.6f}")
        print(f"机械臂2到机械臂1的变换矩阵:")
        print("[[{:.8f}, {:.8f}, {:.8f}, {:.8f}],".format(transform_matrix[0,0], transform_matrix[0,1], transform_matrix[0,2], transform_matrix[0,3]))
        print(" [{:.8f}, {:.8f}, {:.8f}, {:.8f}],".format(transform_matrix[1,0], transform_matrix[1,1], transform_matrix[1,2], transform_matrix[1,3]))
        print(" [{:.8f}, {:.8f}, {:.8f}, {:.8f}],".format(transform_matrix[2,0], transform_matrix[2,1], transform_matrix[2,2], transform_matrix[2,3]))
        print(" [{:.8f}, {:.8f}, {:.8f}, {:.8f}]]".format(transform_matrix[3,0], transform_matrix[3,1], transform_matrix[3,2], transform_matrix[3,3]))
        print(f"机械臂2到机械臂1的RPY角度: {rpy}")

        # 保存标定结果到文本文件
        save_dual_robot_calibration_to_txt(
            config_data['paths']['calibration_results'],
            intrinsic_matrix, distortion_coeffs,
            transform_matrix, inv_transform_matrix, rpy, inv_rpy,
            reprojection_error, dual_robot_error
        )

    except Exception as e:
        print(f"双机械臂标定过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

