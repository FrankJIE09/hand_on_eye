import os
import shutil
import time
import numpy as np
import cv2
from tqdm import tqdm
from ROBOT.dazu.CPS import CPSClient
from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect


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



def capture_image(camera):
    frame_2d = Frame2D()
    camera.capture_2d(frame_2d)
    
    if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
        image = frame_2d.get_gray_scale_image()
        if image is not None:
            return cv2.cvtColor(image.data(), cv2.COLOR_GRAY2BGR)
    elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
        image = frame_2d.get_color_image()
        if image is not None:
            return cv2.cvtColor(image.data(), cv2.COLOR_RGB2BGR)
    return None


# 创建MechEye相机对象
camera = Camera()

# 连接相机
if not find_and_connect(camera):
    print("相机连接失败")
    exit()

cps = CPSClient()
boxID = 0
rbtID = 0
cps.HRIF_Connect(boxID, '172.16.2.101', 10003)

positions = np.loadtxt('./positions.csv', delimiter=',')

# Directory for saving images and pose data
image_dir = './captured_images/'
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)  # Remove the directory if it exists
os.makedirs(image_dir, exist_ok=True)  # Create the directory

pose_data = []

for i, position in enumerate(tqdm(positions, desc="Capturing images and poses")):
    move_robot(cps, boxID, rbtID, position)
    time.sleep(1)

    # 捕获图像
    image = capture_image(camera)
    if image is not None:
        cv2.imwrite(f'{image_dir}image_{i}.png', image)
        current_pose = []
        cps.HRIF_ReadActPos(boxID, rbtID, current_pose)
        current_pose = [float(num) for num in current_pose]
        pose_data.append(current_pose[6:12])
        np.save(f'{image_dir}pose_{i}.npy', np.array(current_pose))

# 断开相机连接
camera.disconnect()
cps.HRIF_DisConnect(boxID)
np.save('./pose_data.npy', np.array(pose_data))
