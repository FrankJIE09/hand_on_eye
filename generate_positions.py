import numpy as np
import cv2
import pyrealsense2 as rs
from GraspErzi.ROBOT.eage_robot_client import RobotClient

def save_positions(positions, filename="positions.csv"):
    """
    将记录的位置保存到 CSV 文件中。
    """
    np.savetxt(filename, positions, delimiter=',', fmt='%.3f')

# 初始化RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动摄像头流
pipeline.start(config)

# 初始化机械臂
robot = RobotClient("ws://192.168.10.1:1880")
robot.connect()

positions = []  # 用于存储记录的位置

print("按下 'r' 键记录当前位置，按下 'q' 键退出并保存位置。")

while True:
    # 获取RealSense帧
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        continue

    # 将图像转换为numpy数组
    color_image = np.asanyarray(color_frame.get_data())

    # 显示图像
    cv2.imshow('RealSense', color_image)

    # 等待用户按键
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        # 获取当前机械臂位置
        current_position = robot.get_geom(matrix=False)
        positions.append(current_position)
        print(f"记录当前位置: {current_position}")
        print(f"已保存位置数: {len(positions)}")

    elif key == ord('q'):
        print("退出并保存位置。")
        break

# 关闭RealSense摄像头流
pipeline.stop()

# 保存所有记录的位置
if positions:
    save_positions(np.array(positions))
    print(f"位置已保存到 'positions.csv' 文件中。共保存了 {len(positions)} 个位置。")

cv2.destroyAllWindows()  # 确保所有OpenCV窗口被关闭
