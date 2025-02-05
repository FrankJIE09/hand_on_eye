import os
import shutil
import time
import numpy as np
import cv2
from tqdm import tqdm
from ROBOT.elibot.CPS import CPSClient
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from utils import frame_to_bgr_image  # 这个utils库可能包含一些辅助函数，这里用于将帧转换为图像


def move_robot(cps_client, target_pose):
    speed = 5  # 运动速度
    cps_client.move_robot(target_pose=target_pose, speed=speed)


def capture_image(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    image = np.asanyarray(color_frame.get_data())
    return image


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
IP = '192.168.11.8'
cps_client = CPSClient(IP)
ret = cps_client.connect()
if ret != True:
    print(f"连接机器人失败，错误码: {ret}")
    exit()
positions = np.loadtxt('./positions.csv', delimiter=',')

# Directory for saving images and pose data
image_dir = './captured_images/'
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)  # Remove the directory if it exists
os.makedirs(image_dir, exist_ok=True)  # Create the directory

pose_data = []

for i, position in enumerate(tqdm(positions, desc="Capturing images and poses")):
    cps_client.move_robot(target_pose=position)

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
    image = color_image
    if image is not None:
        cv2.imwrite(f'{image_dir}image_{i}.png', image)
        current_pose = cps_client.getTCPPose()
        current_pose = [float(num) for num in current_pose]
        pose_data.append(current_pose)
        np.save(f'{image_dir}pose_{i}.npy', np.array(current_pose))

pipeline.stop()
cps_client.disconnect()
np.save('./pose_data.npy', np.array(pose_data))
