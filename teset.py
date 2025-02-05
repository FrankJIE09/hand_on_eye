# 导入必要的模块
import time  # 用于时间相关操作
import cv2  # 用于处理图像的OpenCV库
import numpy as np  # 用于数值计算和数组操作

# 从pyorbbecsdk模块中导入Config、OBSensorType、Pipeline
from pyorbbecsdk import Config  # 用于配置传感器
from pyorbbecsdk import OBSensorType  # 定义传感器类型
from pyorbbecsdk import Pipeline  # 用于管理数据流

# 定义一些常量
ESC_KEY = 27  # 键盘ESC键的ASCII码
PRINT_INTERVAL = 1  # 打印时间间隔（秒）
MIN_DEPTH = 20  # 最小深度阈值（毫米）
MAX_DEPTH = 10000  # 最大深度阈值（毫米）

# 定义一个时间滤波器类，用于对连续帧进行加权平均
class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha  # 滤波器的权重系数，决定当前帧和上一帧的影响比例
        self.previous_frame = None  # 保存上一帧的深度数据

    def process(self, frame):
        if self.previous_frame is None:  # 如果是第一帧，直接返回当前帧
            result = frame
        else:
            # 对当前帧和上一帧按alpha加权平均
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result  # 更新上一帧数据
        return result  # 返回滤波后的结果

# 定义主函数
def main():
    config = Config()  # 创建配置对象
    pipeline = Pipeline()  # 创建数据流对象
    temporal_filter = TemporalFilter(alpha=0.5)  # 创建时间滤波器对象，alpha设置为0.5

    try:
        # 获取深度传感器的流配置
        profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        assert profile_list is not None  # 确保配置列表不为空
        depth_profile = profile_list.get_default_video_stream_profile()  # 获取默认的深度流配置
        assert depth_profile is not None  # 确保配置有效
        print("depth profile: ", depth_profile)  # 打印深度配置
        config.enable_stream(depth_profile)  # 启用该深度流配置
    except Exception as e:  # 捕获异常
        print(e)  # 打印异常信息
        return  # 终止程序

    pipeline.start(config)  # 启动数据流
    last_print_time = time.time()  # 记录当前时间

    while True:  # 主循环
        try:
            frames = pipeline.wait_for_frames(100)  # 等待并获取帧数据，超时时间为100ms
            if frames is None:  # 如果没有获取到帧，继续下一轮循环
                continue
            depth_frame = frames.get_depth_frame()  # 提取深度帧
            if depth_frame is None:  # 如果深度帧为空，继续下一轮循环
                continue

            # 获取深度帧的宽度、高度和深度缩放比例
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()

            # 将深度帧数据转换为NumPy数组
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((height, width))  # 重塑为二维数组

            # 将深度数据转换为浮点数并按缩放比例调整
            depth_data = depth_data.astype(np.float32) * scale
            # 根据深度范围过滤数据，超出范围的设为0
            depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
            depth_data = depth_data.astype(np.uint16)  # 转换回16位整数

            # 应用时间滤波器
            depth_data = temporal_filter.process(depth_data)

            # 获取图像中心点的深度值
            center_y = int(height / 2)  # 计算中心点的y坐标
            center_x = int(width / 2)  # 计算中心点的x坐标
            center_distance = depth_data[center_y, center_x]  # 获取中心点深度值

            # 打印中心点深度值，每PRINT_INTERVAL秒打印一次
            current_time = time.time()
            if current_time - last_print_time >= PRINT_INTERVAL:
                print("center distance: ", center_distance)
                last_print_time = current_time

            # 将深度数据归一化到0-255范围并转换为可视化图像
            depth_image = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)  # 应用伪彩色映射

            cv2.imshow("Depth Viewer", depth_image)  # 显示深度图像窗口
            key = cv2.waitKey(1)  # 等待键盘输入，刷新窗口
            if key == ord('q') or key == ESC_KEY:  # 按'q'或ESC键退出
                break
        except KeyboardInterrupt:  # 捕获键盘中断信号
            break

    pipeline.stop()  # 停止数据流

# 如果脚本是主程序，则执行main函数
if __name__ == "__main__":
    main()
