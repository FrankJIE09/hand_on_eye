import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R


def find_corners(images, pattern_size):
    """
    查找每张图像中的圆形网格角点，并在图像中绘制这些角点。
    返回世界坐标和图像坐标对。
    """
    world_points = create_world_points(pattern_size)
    obj_points = []
    img_points = []

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findCirclesGrid(gray, pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret:
            print(f"i: {i}")
            obj_points.append(world_points)
            img_points.append(corners)
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('findCorners', 640, 480)
            cv2.imshow('findCorners', img)
            cv2.waitKey(200)

    cv2.destroyAllWindows()
    return obj_points, img_points


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


def load_robot_poses(num_poses):
    """
    加载并解析机器人位姿数据文件，提取旋转矩阵和平移向量。
    """
    robot_rot_matrices = []
    robot_trans_vectors = []
    for i in range(num_poses):
        data = np.load(f'./erzhi_T_data_circle2/T_base2ee0{i}.npz')
        transform_matrix = data['arr_0']
        rot_matrix = transform_matrix[:3, :3]
        trans_vector = transform_matrix[:3, 3]
        robot_rot_matrices.append(rot_matrix)
        robot_trans_vectors.append(trans_vector)
    return robot_rot_matrices, robot_trans_vectors


def calibrate_camera(obj_points, img_points, img_size):
    """
    使用OpenCV的calibrateCamera函数进行相机标定，得到内参数矩阵和畸变系数。
    """
    ret, intrinsic_matrix, distortion_coeffs, rot_vectors, trans_vectors = cv2.calibrateCamera(obj_points, img_points,
                                                                                               img_size, None, None)
    optimal_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coeffs, img_size, 0, img_size)
    return ret, intrinsic_matrix, distortion_coeffs, optimal_matrix, rot_vectors, trans_vectors


def hand_eye_calibration(robot_rot_matrices, robot_trans_vectors, cam_rot_matrices, cam_trans_vectors):
    """
    使用TSAI和PARK方法进行手眼标定，得到相机到手眼的变换矩阵。
    """
    rm_tsa, tm_tsa = cv2.calibrateHandEye(robot_rot_matrices, robot_trans_vectors, cam_rot_matrices, cam_trans_vectors,
                                          method=cv2.CALIB_HAND_EYE_TSAI)
    transform_matrix_tsa = create_transformation_matrix(rm_tsa, tm_tsa)

    rm_park, tm_park = cv2.calibrateHandEye(robot_rot_matrices, robot_trans_vectors, cam_rot_matrices,
                                            cam_trans_vectors, method=cv2.CALIB_HAND_EYE_PARK)
    transform_matrix_park = create_transformation_matrix(rm_park, tm_park)

    return transform_matrix_tsa, transform_matrix_park


def create_transformation_matrix(rotation_matrix, translation_vector):
    """
    创建4x4的变换矩阵。
    """
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translation_vector.reshape(-1)
    return transformation_matrix


def rotation_matrix_to_rpy(rotation_matrix):
    """
    转换旋转矩阵到欧拉角 (RPY)。
    """
    r = R.from_matrix(rotation_matrix)
    return r.as_euler('xyz', degrees=True)


def save_calibration_results(filename, ret, intrinsic_matrix, distortion_coeffs, rot_vectors, trans_vectors,
                             optimal_matrix, transform_matrix_tsa, transform_matrix_park, rpy_tsa, rpy_park):
    """
    将标定结果保存到文件。
    """
    with open(filename, 'w') as f:
        f.write("相机标定结果:\n")
        f.write(f"ret: {ret}\n")
        f.write(f"内参数矩阵 (intrinsic_matrix):\n{intrinsic_matrix}\n")
        f.write(f"畸变系数 (distortion_coeffs):\n{distortion_coeffs}\n")

        f.write(f"优化后的相机矩阵 (optimal_matrix):\n{optimal_matrix}\n")

        f.write("\n手眼标定结果 (TSAI 方法):\n")
        f.write(f"transform_matrix_tsa:\n{transform_matrix_tsa}\n")
        f.write(f"RPY_TSAI (度):\n{rpy_tsa}\n")

        f.write("\n手眼标定结果 (PARK 方法):\n")
        f.write(f"transform_matrix_park:\n{transform_matrix_park}\n")
        f.write(f"RPY_PARK (度):\n{rpy_park}\n")

        f.write(f"旋转向量 (rot_vectors):\n{rot_vectors}\n")
        f.write(f"平移向量 (trans_vectors):\n{trans_vectors}\n")
    print(f"手眼标定结果已保存到 {filename}")


def main():
    # 设置参数
    pattern_size = (4, 11)
    num_poses = 48

    # 加载图像
    images = glob.glob('./erzhi_img_circle2/*.jpg')

    # 查找棋盘角点
    obj_points, img_points = find_corners(images, pattern_size)

    # 加载机器人位姿
    robot_rot_matrices, robot_trans_vectors = load_robot_poses(num_poses)

    # 相机标定
    img_size = cv2.imread(images[0]).shape[::-1][1:3]
    ret, intrinsic_matrix, distortion_coeffs, optimal_matrix, rot_vectors, trans_vectors = calibrate_camera(obj_points,
                                                                                                            img_points,
                                                                                                            img_size)

    # 转换旋转向量为旋转矩阵
    cam_rot_matrices = [cv2.Rodrigues(rot_vec)[0] for rot_vec in rot_vectors]

    # 手眼标定
    transform_matrix_tsa, transform_matrix_park = hand_eye_calibration(robot_rot_matrices, robot_trans_vectors,
                                                                       cam_rot_matrices, trans_vectors)

    # 转换旋转矩阵到欧拉角 (RPY)
    rpy_tsa = rotation_matrix_to_rpy(transform_matrix_tsa[:3, :3])
    rpy_park = rotation_matrix_to_rpy(transform_matrix_park[:3, :3])

    # 保存标定结果
    save_calibration_results('hand_eye_calibration_results.txt', ret, intrinsic_matrix, distortion_coeffs, rot_vectors,
                             trans_vectors, optimal_matrix, transform_matrix_tsa, transform_matrix_park, rpy_tsa,
                             rpy_park)


if __name__ == "__main__":
    main()
