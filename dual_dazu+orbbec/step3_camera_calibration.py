import cv2
import numpy as np
import glob
import os
from scipy.spatial.transform import Rotation as R
import re
import yaml
from tqdm import tqdm


def load_config(config_file="config.yaml"):
    """
    加载配置文件
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


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
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rot_vectors[i], trans_vectors[i], intrinsic_matrix,
                                          distortion_coeffs)
        error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    reprojection_error = mean_error / len(obj_points)

    return ret, intrinsic_matrix, distortion_coeffs, optimal_matrix, trans_vectors, cam_rot_matrices, reprojection_error


def save_camera_calibration_results(filename, intrinsic_matrix, distortion_coeffs, 
                                   trans_vectors, cam_rot_matrices, reprojection_error, used_indices):
    """
    保存相机标定结果
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Camera Calibration Results\n")
        f.write("==========================\n\n")
        
        f.write("Camera Matrix (Intrinsic):\n")
        np.savetxt(f, intrinsic_matrix, fmt='%f')
        f.write("\nDistortion Coefficients:\n")
        np.savetxt(f, distortion_coeffs, fmt='%f')
        
        f.write(f"\nReprojection Error: {reprojection_error:.6f} pixels\n")
        f.write(f"Number of Valid Images: {len(used_indices)}\n")
        
        f.write("\nCamera to Board Transformations:\n")
        f.write("Format: [translation_x, translation_y, translation_z, rotation_matrix_3x3]\n")
        for i, (trans_vec, rot_mat) in enumerate(zip(trans_vectors, cam_rot_matrices)):
            f.write(f"\nImage {i:03d}:\n")
            f.write("Translation: ")
            np.savetxt(f, trans_vec.reshape(1, -1), fmt='%f')
            f.write("Rotation Matrix:\n")
            np.savetxt(f, rot_mat, fmt='%f')
    
    # 保存为numpy格式便于后续使用
    np.savez(filename.replace('.txt', '.npz'),
             intrinsic_matrix=intrinsic_matrix,
             distortion_coeffs=distortion_coeffs,
             trans_vectors=trans_vectors,
             cam_rot_matrices=cam_rot_matrices,
             reprojection_error=reprojection_error,
             used_indices=used_indices)
    
    print(f"相机标定结果已保存到 {filename}")
    print(f"相机标定数据已保存到 {filename.replace('.txt', '.npz')}")


def sort_images(images):
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else 0

    return sorted(images, key=extract_number)


def main():
    # 加载配置
    config_data = load_config()

    pattern_size = tuple(config_data['calibration_board']['pattern_size'])
    images = glob.glob(f"{config_data['paths']['images_dir']}*.png")

    if not images:
        print(f"错误：在 {config_data['paths']['images_dir']} 目录中未找到图片文件")
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
        # 相机标定
        img_size = cv2.imread(images[0]).shape[::-1][1:3]
        ret, intrinsic_matrix, distortion_coeffs, optimal_matrix, trans_vectors, cam_rot_matrices, reprojection_error = calibrate_camera(
            obj_points, img_points, img_size)

        print(f"相机标定重投影误差: {reprojection_error:.6f} 像素")
        print(f"相机内参矩阵:")
        print(intrinsic_matrix)
        print(f"畸变系数:")
        print(distortion_coeffs)

        # 保存相机标定结果
        save_camera_calibration_results(
            "camera_calibration_results.txt",
            intrinsic_matrix, distortion_coeffs,
            trans_vectors, cam_rot_matrices, reprojection_error, used_indices
        )

    except Exception as e:
        print(f"相机标定过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


