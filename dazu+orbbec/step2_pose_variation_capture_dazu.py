import os
import shutil
import time
import numpy as np
import cv2
from tqdm import tqdm
from ROBOT.dazu.CPS import CPSClient
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from utils import frame_to_bgr_image  # 这个utils库可能包含一些辅助函数，这里用于将帧转换为图像


def move_robot(cps_client, boxID, rbtID, target_pose):
    speed = 50  # 运动速度
    acceleration = 500  # 加速度
    ucs = "Base"  # 坐标系
    radius = 0  # 直线运动半径
    ret = cps_client.HRIF_MoveL(boxID, rbtID, points=target_pose, RawACSpoints=target_pose, tcp="TCP", ucs=ucs,
                                speed=speed, Acc=acceleration, radius=radius, isSeek=0, bit=0, state=1, cmdID=1)
    if ret == 0:
        # 等待运动完成
        while True:
            motion_done_result = []
            motion_done = cps_client.HRIF_IsMotionDone(boxID, rbtID, motion_done_result)
            if motion_done == 0 and motion_done_result and motion_done_result[0] == 1:
                break  # 运动完成
            elif motion_done < 0:
                print(f"运动过程中出错，错误码: {motion_done}")
                break
            time.sleep(0.5)  # 等待一段时间再次检查
    else:
        print(f"机器人运动失败，错误码: {ret}")



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
        color_profile = profile_list.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
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

cps = CPSClient()
boxID = 0
rbtID = 0
cps.HRIF_Connect(boxID, '192.168.188.102', 10003)

positions = np.loadtxt('./positions.csv', delimiter=',')

# Directory for saving images and pose data
image_dir = './captured_images/'
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)  # Remove the directory if it exists
os.makedirs(image_dir, exist_ok=True)  # Create the directory

pose_data = []

for i, position in enumerate(tqdm(positions, desc="Capturing images and poses")):
    move_robot(cps, boxID, rbtID, position)

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
        current_pose = []
        cps.HRIF_ReadActPos(boxID, rbtID, current_pose)
        current_pose = [float(num) for num in current_pose]
        pose_data.append(current_pose[6:12])
        np.save(f'{image_dir}pose_{i}.npy', np.array(current_pose))

pipeline.stop()
cps.HRIF_DisConnect(boxID)
np.save('./pose_data.npy', np.array(pose_data))
