import numpy as np
import cv2
import time
import random
import yaml
from ROBOT.dazu.CPS import CPSClient

from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from utils import frame_to_bgr_image  # 这个utils库可能包含一些辅助函数，这里用于将帧转换为图像

ESC_KEY = 27


def load_config(config_file="config.yaml"):
    """
    加载配置文件
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_positions(robot1_positions, robot2_positions, filename="positions.csv"):
    """
    将记录的双机械臂位置保存到 CSV 文件中。
    格式: robot1_x, robot1_y, robot1_z, robot1_rx, robot1_ry, robot1_rz, robot2_x, robot2_y, robot2_z, robot2_rx, robot2_ry, robot2_rz
    """
    combined_positions = np.hstack([robot1_positions, robot2_positions])
    np.savetxt(filename, combined_positions, delimiter=',', fmt='%.3f')



# 加载配置
config_data = load_config()

# 相机配置
camera_config = Config()  # 创建配置对象
pipeline = Pipeline()  # 创建流管道，用于处理来自相机的数据流

try:
    # 获取可用的视频流配置列表
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        # 尝试获取指定分辨率和帧率的RGB视频流配置
        color_profile = profile_list.get_video_stream_profile(
            config_data['camera']['width'], 
            config_data['camera']['height'], 
            OBFormat.RGB, 
            config_data['camera']['fps']
        )
    except OBError as e:
        # 如果指定配置失败，则获取默认的视频流配置
        print(e)
        color_profile = profile_list.get_default_video_stream_profile()
        print("color profile: ", color_profile)
    # 启用视频流配置
    camera_config.enable_stream(color_profile)
except Exception as e:
    # 如果配置流过程中出错，则输出错误并返回
    print(e)

# 启动管道，开始处理数据流
pipeline.start(camera_config)

# 连接双机械臂
robot1_config = config_data['robot1']
robot2_config = config_data['robot2']

# 机械臂1 (安装相机)
cps_client1 = CPSClient()
ret1 = cps_client1.HRIF_Connect(robot1_config['box_id'], robot1_config['ip'], robot1_config['port'])
if ret1 != 0:
    print(f"连接机械臂1失败，错误码: {ret1}")
    exit()

# 机械臂2 (安装标定板)
cps_client2 = CPSClient()
ret2 = cps_client2.HRIF_Connect(robot2_config['box_id'], robot2_config['ip'], robot2_config['port'])
if ret2 != 0:
    print(f"连接机械臂2失败，错误码: {ret2}")
    exit()

robot1_positions = []  # 用于存储机械臂1的位置
robot2_positions = []  # 用于存储机械臂2的位置

print("双机械臂位置采集模式")
print("按下 'r' 键记录当前双机械臂位置")
print("按下 'q' 键退出并保存位置")
print("机械臂1: 安装相机")
print("机械臂2: 安装标定板")

while True:
    try:
        # 等待新的帧集合，超时时间为100毫秒
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        # 获取颜色帧
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        # 转换帧为BGR格式的图像（适用于OpenCV处理）
        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            print("failed to convert frame to image")
            continue
    except KeyboardInterrupt:
        # 如果用户使用键盘中断（通常是Ctrl+C），也退出循环
        break

    cv2.imshow('Dual Robot Position Capture', color_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        # 读取机械臂1位置
        result1 = []
        error_code1 = cps_client1.HRIF_ReadActPos(robot1_config['box_id'], robot1_config['robot_id'], result1)
        result1 = [float(num) for num in result1]
        robot1_pos = result1[6:12]
        
        # 读取机械臂2位置
        result2 = []
        error_code2 = cps_client2.HRIF_ReadActPos(robot2_config['box_id'], robot2_config['robot_id'], result2)
        result2 = [float(num) for num in result2]
        robot2_pos = result2[6:12]
        
        if error_code1 == 0 and error_code2 == 0:
            robot1_positions.append(robot1_pos)
            robot2_positions.append(robot2_pos)
            print(f"记录机械臂1位置: {robot1_pos}")
            print(f"记录机械臂2位置: {robot2_pos}")
            print(f"已保存位置对数: {len(robot1_positions)}")
        else:
            print(f"读取位置失败 - 机械臂1错误码: {error_code1}, 机械臂2错误码: {error_code2}")
    elif key == ord('q'):
        print("退出并保存位置。")
        break

pipeline.stop()
if robot1_positions and robot2_positions:
    save_positions(np.array(robot1_positions), np.array(robot2_positions))
    print(f"双机械臂位置已保存到 '{config_data['paths']['positions_file']}' 文件中。")
    print(f"共保存了 {len(robot1_positions)} 个位置对。")

cv2.destroyAllWindows()  # 确保所有OpenCV窗口被关闭
cps_client1.HRIF_DisConnect(robot1_config['box_id'])
cps_client2.HRIF_DisConnect(robot2_config['box_id'])
