import time

import numpy as np
import cv2
import pyrealsense2 as rs
from GraspErzi.ROBOT.eage_robot_client import RobotClient


# Function to initialize Realsense Camera
def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


# Function to capture an image from the Realsense camera
def capture_image(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    image = np.asanyarray(color_frame.get_data())
    return image


# Initialize robotic arm
robot = RobotClient("ws://192.168.10.1:1880")
robot.connect()

# Initialize Realsense camera
pipeline = initialize_realsense()

# Load positions from a file or define them here
positions = np.loadtxt('positions.csv', delimiter=',')

# Directory for saving images and pose data
image_dir = './captured_images/'
pose_data = []

# Move the robot and capture images
for i, position in enumerate(positions):
    robot.move_line(position)
    time.sleep(0.5)
    image = capture_image(pipeline)

    if image is not None:
        # Save the image
        cv2.imwrite(f'{image_dir}image_{i}.png', image)
        # Fetch and save the current pose
        current_pose = robot.get_geom(matrix=False)  # Or however you fetch the pose
        pose_data.append(current_pose)

        # Optionally, save current pose data to a file immediately
        np.save(f'{image_dir}pose_{i}.npy', np.array(current_pose))

# Clean up
pipeline.stop()
robot.close()

# Save all poses to a file
np.save('pose_data.npy', np.array(pose_data))
