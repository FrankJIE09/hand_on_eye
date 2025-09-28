import numpy as np
import cv2
import time
import random
import yaml
from ROBOT.dazu.CPS import CPSClient

from mecheye.shared import *
from mecheye.area_scan_3d_camera import *
from mecheye.area_scan_3d_camera_utils import find_and_connect

ESC_KEY = 27

# 图像显示设置
DISPLAY_SCALE_FACTOR = 0.3  # 显示缩放因子，可以调整这个值来改变显示大小



def save_positions(positions, filename="positions.csv"):
    """
    将记录的位置保存到 CSV 文件中。
    """
    np.savetxt(filename, positions, delimiter=',', fmt='%.3f')



# 创建MechEye相机对象
camera = Camera()

# 连接相机
if not find_and_connect(camera):
    print("相机连接失败")
    exit()

# 设置相机分辨率（减小图像尺寸）
try:
    user_set = camera.current_user_set()
    
    # 获取可用的分辨率
    resolutions = CameraResolutions()
    camera.get_camera_resolutions(resolutions)
    print(f"可用分辨率: 2D={resolutions.color_width()}x{resolutions.color_height()}, 3D={resolutions.depth_width()}x{resolutions.depth_height()}")
    
    # 尝试设置较小的分辨率（如果支持的话）
    # 注意：不是所有相机都支持动态分辨率调整
    print("尝试设置较小的分辨率...")
    
except Exception as e:
    print(f"设置分辨率时出错: {e}")
    print("使用默认分辨率")

# 机器人连接参数
IP = '172.16.2.101'
PORT = 10003
cps_client = CPSClient()
ret = cps_client.HRIF_Connect(0, IP, PORT)
if ret != 0:
    print(f"连接机器人失败，错误码: {ret}")
    exit()

positions = []  # 用于存储记录的位置

print("按下 'r' 键记录当前位置，按下 'q' 键退出并保存位置。")
print(f"图像显示缩放因子: {DISPLAY_SCALE_FACTOR} (原始图像会缩小到 {DISPLAY_SCALE_FACTOR*100:.0f}%)")

while True:
    try:
        # 捕获2D图像
        frame_2d = Frame2D()
        camera.capture_2d(frame_2d)
        
        # 获取图像数据
        if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
            image = frame_2d.get_gray_scale_image()
            if image is not None:
                color_image = cv2.cvtColor(image.data(), cv2.COLOR_GRAY2BGR)
            else:
                continue
        elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
            image = frame_2d.get_color_image()
            if image is not None:
                color_image = cv2.cvtColor(image.data(), cv2.COLOR_RGB2BGR)
            else:
                continue
        else:
            continue
    except KeyboardInterrupt:
        # 如果用户使用键盘中断（通常是Ctrl+C），也退出循环
        break

    # 缩放图像以减小显示尺寸
    height, width = color_image.shape[:2]
    new_width = int(width * DISPLAY_SCALE_FACTOR)
    new_height = int(height * DISPLAY_SCALE_FACTOR)
    resized_image = cv2.resize(color_image, (new_width, new_height))
    
    cv2.imshow('mecheye', resized_image)

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

# 断开相机连接
camera.disconnect()

if positions:
    save_positions(np.array(positions))
    print(f"位置已保存到 'positions.csv' 文件中。共保存了 {len(positions)} 个位置。")

cv2.destroyAllWindows()  # 确保所有OpenCV窗口被关闭
cps_client.HRIF_DisConnect(0)
