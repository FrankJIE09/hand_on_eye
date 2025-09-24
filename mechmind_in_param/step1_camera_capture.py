#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
梅卡相机拍照脚本
用于标定板图像采集
"""

import cv2
import os
import time
from datetime import datetime

try:
    from mecheye.shared import *
    from mecheye.area_scan_3d_camera import *
    from mecheye.area_scan_3d_camera_utils import find_and_connect
    MECHEYE_AVAILABLE = True
except ImportError:
    print("警告：梅卡相机SDK未安装，无法使用相机功能")
    MECHEYE_AVAILABLE = False


class MechEyeCameraCapture:
    """梅卡相机拍照类"""
    
    def __init__(self):
        self.camera = None
        self.capture_count = 0
        self.captured_images_dir = "./captured_images"
        
        # 创建保存目录
        os.makedirs(self.captured_images_dir, exist_ok=True)
    
    def connect_camera(self):
        """连接相机"""
        if not MECHEYE_AVAILABLE:
            print("错误：梅卡相机SDK未安装")
            return False
            
        try:
            self.camera = Camera()
            if find_and_connect(self.camera):
                print("成功连接到梅卡相机")
                return True
            else:
                print("无法连接到梅卡相机")
                return False
        except Exception as e:
            print(f"连接相机时出错: {e}")
            return False
    
    def capture_image(self):
        """拍摄单张图像"""
        if not MECHEYE_AVAILABLE or self.camera is None:
            print("错误：相机未连接")
            return None
            
        try:
            # 捕获2D图像
            frame_2d = Frame2D()
            show_error(self.camera.capture_2d(frame_2d))
            
            # 获取图像数据
            if frame_2d.color_type() == ColorTypeOf2DCamera_Monochrome:
                image2d = frame_2d.get_gray_scale_image()
                # 转换为彩色图像用于显示
                rgb_image = cv2.cvtColor(image2d.data(), cv2.COLOR_GRAY2BGR)
            elif frame_2d.color_type() == ColorTypeOf2DCamera_Color:
                image2d = frame_2d.get_color_image()
                # 转换为BGR格式（OpenCV默认格式）
                rgb_image = cv2.cvtColor(image2d.data(), cv2.COLOR_RGB2BGR)
            else:
                print("警告：未知的图像类型")
                return None
                
            return rgb_image
            
        except Exception as e:
            print(f"拍摄图像时出错: {e}")
            return None
    
    def save_image(self, image, filename=None):
        """保存图像到文件"""
        if image is None:
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
            filename = f"{self.capture_count:03d}_{timestamp}.png"
        
        filepath = os.path.join(self.captured_images_dir, filename)
        
        try:
            cv2.imwrite(filepath, image)
            print(f"图像已保存: {filepath}")
            self.capture_count += 1
            return filepath
        except Exception as e:
            print(f"保存图像时出错: {e}")
            return None
    
    def run_interactive_mode(self):
        """交互式拍照模式"""
        print("=== 梅卡相机标定板拍照工具 ===")
        print("按 's' 键拍摄图像")
        print("按 'q' 键退出程序")
        print("=" * 40)
        
        # 连接相机
        if not self.connect_camera():
            print("无法连接相机，程序退出")
            return
        
        # 创建窗口
        cv2.namedWindow("梅卡相机实时图像", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("梅卡相机实时图像", 800, 600)
        
        print("开始实时显示...")
        
        while True:
            # 捕获图像
            rgb_image = self.capture_image()
            
            if rgb_image is not None:
                # 显示图像
                cv2.imshow("梅卡相机实时图像", rgb_image)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("用户按下q键，程序退出")
                break
            elif key == ord('s'):
                print("用户按下s键，拍摄图像...")
                
                # 保存当前图像
                saved_path = self.save_image(rgb_image)
                if saved_path:
                    print(f"拍摄成功！已保存 {self.capture_count} 张图像")
                else:
                    print("拍摄失败")
            
            # 短暂延迟
            time.sleep(0.01)
        
        # 清理资源
        cv2.destroyAllWindows()
        if self.camera is not None:
            self.camera.disconnect()
            print("相机已断开连接")
        
        print(f"程序结束，共拍摄了 {self.capture_count} 张图像")
    
    def capture_multiple_images(self, num_images=20, interval=2.0):
        """自动拍摄多张图像"""
        print(f"=== 自动拍摄模式 ===")
        print(f"将拍摄 {num_images} 张图像，间隔 {interval} 秒")
        print("按 Ctrl+C 可提前停止")
        print("=" * 30)
        
        # 连接相机
        if not self.connect_camera():
            print("无法连接相机，程序退出")
            return
        
        try:
            for i in range(num_images):
                print(f"拍摄第 {i+1}/{num_images} 张图像...")
                
                # 捕获图像
                rgb_image = self.capture_image()
                
                if rgb_image is not None:
                    # 保存图像
                    saved_path = self.save_image(rgb_image)
                    if saved_path:
                        print(f"第 {i+1} 张图像拍摄成功")
                    else:
                        print(f"第 {i+1} 张图像拍摄失败")
                else:
                    print(f"第 {i+1} 张图像捕获失败")
                
                # 等待间隔时间（除了最后一张）
                if i < num_images - 1:
                    time.sleep(interval)
            
            print(f"自动拍摄完成！共拍摄了 {self.capture_count} 张图像")
            
        except KeyboardInterrupt:
            print("\n用户中断拍摄")
            print(f"已拍摄 {self.capture_count} 张图像")
        
        finally:
            # 断开相机连接
            if self.camera is not None:
                self.camera.disconnect()
                print("相机已断开连接")


def main():
    """主函数"""
    capture_tool = MechEyeCameraCapture()
    
    print("选择拍照模式:")
    print("1. 交互式拍照 (手动按s键拍照)")
    print("2. 自动拍照 (自动拍摄多张图像)")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            capture_tool.run_interactive_mode()
        elif choice == "2":
            try:
                num_images = int(input("请输入要拍摄的图像数量 (默认20): ") or "20")
                interval = float(input("请输入拍摄间隔时间(秒) (默认2.0): ") or "2.0")
                capture_tool.capture_multiple_images(num_images, interval)
            except ValueError:
                print("输入格式错误，使用默认参数")
                capture_tool.capture_multiple_images()
        else:
            print("无效选择，使用交互式模式")
            capture_tool.run_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")


if __name__ == "__main__":
    main()
