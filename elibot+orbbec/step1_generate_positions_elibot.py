import numpy as np
import cv2
import time
import random
import yaml
from ROBOT.elibot.CPS import CPSClient

from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from utils import frame_to_bgr_image  # 这个utils库可能包含一些辅助函数，这里用于将帧转换为图像

ESC_KEY = 27




def save_positions(positions, filename="positions.csv"):
    """
    将记录的位置保存到 CSV 文件中。
    """
    np.savetxt(filename, positions, delimiter=',', fmt='%.3f')





config = Config()  # 创建配置对象
pipeline = Pipeline()  # 创建流管道，用于处理来自相机的数据流

try:
    # 获取可用的视频流配置列表
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        # 尝试获取指定分辨率和帧率的RGB视频流配置
        color_profile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
    except OBError as e:
        # 如果指定配置失败，则获取默认的视频流配置
        print(e)
        color_profile = profile_list.get_default_video_stream_profile()
        print("color profile: ", color_profile)
    # 启用视频流配置
    config.enable_stream(color_profile)
except Exception as e:
    # 如果配置流过程中出错，则输出错误并返回
    print(e)

# 启动管道，开始处理数据流
pipeline.start(config)

# 机器人连接参数
IP = '192.168.1.201'
cps_client = CPSClient(IP)
ret = cps_client.connect()
if ret != True:
    print(f"连接机器人失败，错误码: {ret}")
    exit()

positions = []  # 用于存储记录的位置

print("按下 'r' 键记录当前位置，按下 'q' 键退出并保存位置。")

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

    cv2.imshow('orbbec', color_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        result = cps_client.getTCPPose()
        result = [float(num) for num in result]
        positions.append(result)
        print(f"记录当前位置: {result}")
        print(f"已保存位置数: {len(positions)}")
    elif key == ord('q'):
        print("退出并保存位置。")
        break

pipeline.stop()
if positions:
    save_positions(np.array(positions))
    print(f"位置已保存到 'positions.csv' 文件中。共保存了 {len(positions)} 个位置。")

cv2.destroyAllWindows()  # 确保所有OpenCV窗口被关闭
cps_client.disconnect()
