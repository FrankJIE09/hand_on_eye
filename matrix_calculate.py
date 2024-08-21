import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R
import re
import yaml
from tqdm import tqdm

def find_corners(images, pattern_size):
    """
    查找每张图像中的圆形网格角点，并在图像中绘制这些角点。
    返回世界坐标和图像坐标对。
    """
    world_points = create_world_points(pattern_size)
    obj_points = []
    img_points = []
    used_indices = []
    unused_images = []

    for i, fname in tqdm(enumerate(images), desc="Finding corners", total=len(images)):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret:
            obj_points.append(world_points)
            img_points.append(corners)
            used_indices.append(i)  # 记录成功检测到角点的图像索引
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            # Save processed image
            output_fname = f"./processed_images/{i:03d}.png"
            cv2.imwrite(output_fname, img)
        else:
            unused_images.append(fname)  # 收集未使用的图片

    return obj_points, img_points, used_indices, unused_images

def create_world_points(pattern_size):
    """
    生成世界坐标系中的圆心点。
    """
    width, height = pattern_size
    world_points = np.zeros((width * height, 3), np.float32)
    num = 0
    for i in range(height):
        for j in range(width):
            world_points[num, :2] = [j + 0.5 * (i % 2), i * 0.5]
            num += 1
    return world_points * 0.02

def load_robot_poses(file_path, used_indices):
    """
    从文件加载并解析机器人位姿数据，提取旋转矩阵和平移向量。
    只加载与成功检测到角点的图像对应的姿态。
    """
    data = np.load(file_path, allow_pickle=True)

    robot_rot_matrices = []
    robot_trans_vectors = []

    valid_indices = [i for i in used_indices if i < len(data)]  # 确保索引不超范围

    for i in tqdm(valid_indices, desc="Loading robot poses"):
        pose = data[i]
        if len(pose) == 6:
            # 前三个是平移向量
            trans_vector = np.array(pose[:3]) / 1000  # 假设平移单位为毫米，将其转换为米
            # 后三个是旋转向量 (假设是欧拉角)
            rot_vector = np.array(pose[3:])

            # 将旋转向量转换为旋转矩阵
            rotation_matrix = R.from_euler('xyz', rot_vector, degrees=True).as_matrix()

            robot_rot_matrices.append(rotation_matrix)
            robot_trans_vectors.append(trans_vector)
        else:
            raise ValueError("每个pose数据应包含6个元素（3个平移+3个旋转）")

    return robot_rot_matrices, robot_trans_vectors

def calibrate_camera(obj_points, img_points, img_size):
    """
    使用OpenCV的calibrateCamera函数进行相机标定，得到内参数矩阵和畸变系数。
    """
    ret, intrinsic_matrix, distortion_coeffs, rot_vectors, trans_vectors = cv2.calibrateCamera(obj_points, img_points,
                                                                                               img_size, None, None)
    optimal_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, img_size, 0, img_size)
    # 转换旋转向量为旋转矩阵
    cam_rot_matrices = [cv2.Rodrigues(rot_vec)[0] for rot_vec in rot_vectors]

    return ret, intrinsic_matrix, distortion_coeffs, optimal_matrix, trans_vectors, cam_rot_matrices

def hand_eye_calibration(robot_rot_matrices, robot_trans_vectors, cam_rot_matrices, cam_trans_vectors,
                         method=cv2.CALIB_HAND_EYE_PARK):
    """
    使用指定方法进行手眼标定，得到相机到手眼的变换矩阵，并计算RPY。

    支持的标定方法包括：
    - cv2.CALIB_HAND_EYE_TSAI
    - cv2.CALIB_HAND_EYE_PARK
    - cv2.CALIB_HAND_EYE_HORAUD
    - cv2.CALIB_HAND_EYE_DANIILIDIS
    """
    rm, tm = cv2.calibrateHandEye(robot_rot_matrices, robot_trans_vectors, cam_rot_matrices,
                                  cam_trans_vectors, method=method)
    transform_matrix = create_transformation_matrix(rm, tm)

    # 计算RPY (Roll, Pitch, Yaw)
    rpy = R.from_matrix(rm).as_euler('xyz', degrees=True)

    return transform_matrix, rpy

def create_transformation_matrix(rotation_matrix, translation_vector):
    """
    创建4x4的变换矩阵。
    """
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translation_vector.reshape(-1)
    return transformation_matrix

def save_calibration_to_yaml_and_txt(yaml_filename, txt_filename, intrinsic_matrix, distortion_coeffs,
                                     transform_matrix, rpy):
    """
    将标定结果保存为 YAML 和 TXT 文件。
    """
    # YAML 文件保存
    calibration_data = {
        'camera_matrix': intrinsic_matrix.tolist(),
        'distortion_coefficients': distortion_coeffs.tolist(),
        'hand_eye_transformation_matrix': transform_matrix.tolist(),
        'hand_eye_rpy': rpy.tolist()
    }

    with open(yaml_filename, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=None)
    print(f"标定结果已保存到 {yaml_filename}")

    # TXT 文件保存
    with open(txt_filename, 'w') as f:
        f.write("Camera Matrix (Intrinsic):\n")
        np.savetxt(f, intrinsic_matrix, fmt='%f')

        f.write("\nDistortion Coefficients:\n")
        np.savetxt(f, distortion_coeffs, fmt='%f')

        f.write("\nHand-Eye Transformation Matrix:\n")
        np.savetxt(f, transform_matrix, fmt='%f')

        f.write("\nHand-Eye RPY:\n")
        np.savetxt(f, rpy, fmt='%f')

    print(f"标定结果已保存到 {txt_filename}")

def sort_images(images):
    """
    按照图像文件名中的数字对图像路径进行排序。
    """

    def extract_number(filename):
        # 从文件名中提取数字
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else 0

    # 使用提取的数字对图像路径进行排序
    return sorted(images, key=extract_number)

def main():
    # 设置参数
    pattern_size = (4, 11)
    expected_num_poses = 48

    # 加载图像
    images = glob.glob('./captured_images/*.png')
    images = sort_images(images)  # 按照文件名中的数字排序

    # 查找棋盘角点
    obj_points, img_points, used_indices, unused_images = find_corners(images, pattern_size)

    if unused_images:
        print(f"未处理的图片数量: {len(unused_images)}")
        print(f"未处理的图片: {unused_images}")

    # 加载机器人位姿，只加载成功检测到角点的图像对应的姿态
    robot_rot_matrices, robot_trans_vectors = load_robot_poses('./pose_data.npy', used_indices)

    # 相机标定
    img_size = cv2.imread(images[0]).shape[::-1][1:3]
    ret, intrinsic_matrix, distortion_coeffs, optimal_matrix, trans_vectors, cam_rot_matrices = calibrate_camera(obj_points, img_points,
                                                                                               img_size)

    # 手眼标定
    transform_matrix, rpy = hand_eye_calibration(robot_rot_matrices, robot_trans_vectors,
                                                           cam_rot_matrices, trans_vectors)

    # 保存标定结果到 config.yaml 和 calibration_results.txt
    save_calibration_to_yaml_and_txt('config.yaml', 'calibration_results.txt', intrinsic_matrix, distortion_coeffs,
                                     transform_matrix, rpy)

if __name__ == "__main__":
    main()
