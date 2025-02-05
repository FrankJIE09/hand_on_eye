# 引入必要的库
import cv2
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from utils import frame_to_bgr_image  # 这个utils库可能包含一些辅助函数，这里用于将帧转换为图像

ESC_KEY = 27  # 定义退出键为ESC键

def main():
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
        return
    # 启动管道，开始处理数据流
    pipeline.start(config)

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
            # 显示图像
            cv2.imshow("Color Viewer", color_image)
            # 检测按键，如果按下'q'或ESC键，则退出循环
            key = cv2.waitKey(1)
            if key == ord('q') or key == ESC_KEY:
                break
        except KeyboardInterrupt:
            # 如果用户使用键盘中断（通常是Ctrl+C），也退出循环
            break

    # 停止数据流处理，关闭管道
    pipeline.stop()

if __name__ == "__main__":
    main()  # 运行主函数
