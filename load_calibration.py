import yaml
import numpy as np


def load_calibration_from_yaml(yaml_filename):
    """
    从 YAML 文件中读取标定数据，包括相机内参、畸变系数和手眼标定变换矩阵。
    """
    with open(yaml_filename, 'r') as f:
        calibration_data = yaml.safe_load(f)

    # 读取相机内参
    camera_matrix = np.array(calibration_data['camera_matrix'])

    # 读取畸变系数
    distortion_coefficients = np.array(calibration_data['distortion_coefficients'])

    # 读取手眼标定变换矩阵
    hand_eye_transformation_matrix = np.array(calibration_data['hand_eye_transformation_matrix'])

    return camera_matrix, distortion_coefficients, hand_eye_transformation_matrix


def main():
    # 设置文件名
    yaml_filename = 'config.yaml'

    # 从 YAML 文件中加载标定数据
    camera_matrix, distortion_coefficients, hand_eye_transformation_matrix = load_calibration_from_yaml(yaml_filename)

    # 打印加载的数据
    print("Camera Matrix (Intrinsic):")
    print(camera_matrix)

    print("\nDistortion Coefficients:")
    print(distortion_coefficients)

    print("\nHand-Eye Transformation Matrix (PARK):")
    print(hand_eye_transformation_matrix)


if __name__ == "__main__":
    main()
