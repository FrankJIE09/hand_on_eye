import numpy as np
import cv2
import pyrealsense2 as rs
import time
import random
import yaml
from ROBOT.dazu.CPS import CPSClient


def get_current_position(boxID, rbtID):
    cps_client = CPSClient()
    result = []
    error_code = cps_client.HRIF_ReadActPos(boxID, rbtID, result)
    if error_code == 0:
        print("当前位置读取成功：", result)
    else:
        print("读取失败，错误码：", error_code)


def save_positions(positions, filename="positions.csv"):
    """
    将记录的位置保存到 CSV 文件中。
    """
    np.savetxt(filename, positions, delimiter=',', fmt='%.3f')


# 读取配置文件
def load_config(config_file='config.yaml'):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config.get('target_pose', [])
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return []


# 初始化RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 机器人连接参数
IP = '192.168.11.7'
PORT = 10003
cps_client = CPSClient()
ret = cps_client.HRIF_Connect(0, IP, PORT)
if ret != 0:
    print(f"连接机器人失败，错误码: {ret}")
    exit()

positions = []  # 用于存储记录的位置
target_pose = load_config()

print("按下 'r' 键记录当前位置，按下 'q' 键退出并保存位置。")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow('RealSense', color_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        result = []
        error_code = cps_client.HRIF_ReadActPos(0, 0, result)
        result = [float(num) for num in result]
        positions.append(result[6:12])
        print(f"记录当前位置: {result[6:12]}")
        print(f"已保存位置数: {len(positions)}")
    elif key == ord('q'):
        print("退出并保存位置。")
        break

pipeline.stop()
if positions:
    save_positions(np.array(positions))
    print(f"位置已保存到 'positions.csv' 文件中。共保存了 {len(positions)} 个位置。")

cv2.destroyAllWindows()  # 确保所有OpenCV窗口被关闭
cps_client.HRIF_DisConnect(0)
