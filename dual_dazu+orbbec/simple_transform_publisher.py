#!/usr/bin/env python3
"""
简单的ROS变换发布器
持续发布单位矩阵和指定的变换矩阵
"""

import rospy
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R


def create_transformation_matrix(rotation_matrix, translation_vector):
    """创建4x4变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation_vector.flatten()
    return T


def matrix_to_quaternion(rotation_matrix):
    """将旋转矩阵转换为四元数"""
    r = R.from_matrix(rotation_matrix)
    quat = r.as_quat()  # [x, y, z, w]
    return quat


def publish_transform(tf_broadcaster, parent_frame, child_frame, transform_matrix):
    """发布变换矩阵"""
    transform = TransformStamped()
    
    # 设置header
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = parent_frame
    transform.child_frame_id = child_frame
    
    # 设置平移
    transform.transform.translation.x = transform_matrix[0, 3]
    transform.transform.translation.y = transform_matrix[1, 3]
    transform.transform.translation.z = transform_matrix[2, 3]
    
    # 设置旋转（四元数）
    rotation_matrix = transform_matrix[:3, :3]
    quat = matrix_to_quaternion(rotation_matrix)
    transform.transform.rotation.x = quat[0]
    transform.transform.rotation.y = quat[1]
    transform.transform.rotation.z = quat[2]
    transform.transform.rotation.w = quat[3]
    
    # 发布变换
    tf_broadcaster.sendTransform(transform)


def main():
    # 初始化ROS节点
    rospy.init_node('simple_transform_publisher', anonymous=True)
    
    # 创建TF广播器
    tf_broadcaster = TransformBroadcaster()
    
    # 设置发布频率
    rate = rospy.Rate(10)  # 10Hz
    
    # 您提供的变换矩阵
    transform_matrix = np.array([[0.99996553, -0.00813353, 0.00166601, -0.00036655],
 [-0.00813281, -0.99996683, -0.00043604, 0.00238746],
 [0.00166950, 0.00042248, -0.99999852, -0.10147562],
 [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
    
    identity_matrix = np.eye(4)
    
    print("开始发布变换矩阵...")
    print("单位矩阵:")
    print(identity_matrix)
    print("变换矩阵:")
    print(transform_matrix)
    
    try:
        while not rospy.is_shutdown():
            # 发布单位矩阵 (world -> identity)
            publish_transform(tf_broadcaster, "world", "left_robot", identity_matrix)
            
            # 发布变换矩阵 (identity -> transform)
            publish_transform(tf_broadcaster, "left_robot", "right_robot", transform_matrix)
            
            rate.sleep()
            
    except rospy.ROSInterruptException:
        print("节点被中断")
    except Exception as e:
        print(f"节点运行出错: {e}")


if __name__ == "__main__":
    main()
