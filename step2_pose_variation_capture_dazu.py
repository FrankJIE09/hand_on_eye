import os
import shutil
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from tqdm import tqdm
from ROBOT.dazu.CPS import CPSClient

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

def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

def capture_image(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    image = np.asanyarray(color_frame.get_data())
    return image

cps = CPSClient()
boxID = 0
rbtID = 0
cps.HRIF_Connect(boxID, '192.168.11.7', 10003)
pipeline = initialize_realsense()
positions = np.loadtxt('positions.csv', delimiter=',')

# Directory for saving images and pose data
image_dir = './captured_images/'
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)  # Remove the directory if it exists
os.makedirs(image_dir, exist_ok=True)  # Create the directory

pose_data = []

for i, position in enumerate(tqdm(positions, desc="Capturing images and poses")):
    move_robot(cps, boxID, rbtID, position)
    image = capture_image(pipeline)
    if image is not None:
        cv2.imwrite(f'{image_dir}image_{i}.png', image)
        current_pose = []
        cps.HRIF_ReadActPos(boxID, rbtID, current_pose)
        current_pose = [float(num) for num in current_pose]
        pose_data.append(current_pose[6:12])
        np.save(f'{image_dir}pose_{i}.npy', np.array(current_pose))

pipeline.stop()
cps.HRIF_DisConnect(boxID)
np.save('pose_data.npy', np.array(pose_data))
