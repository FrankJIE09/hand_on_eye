import numpy as np
import random
from GraspErzi.ROBOT.eage_robot_client import RobotClient


def generate_random_positions(base_position, num_positions, translation_range, rotation_range):
    """
    Generate a list of random positions based on a base position.

    Parameters:
    - base_position: Tuple or list of the form (x, y, z, rx, ry, rz)
    - num_positions: Number of random positions to generate
    - translation_range: Max deviation allowed in translation (x, y, z)
    - rotation_range: Max deviation allowed in rotation (rx, ry, rz)

    Returns:
    - Array of positions
    """
    positions = []
    print("Base Position:", base_position)  # Debugging output

    if len(base_position) != 6:
        raise ValueError("Base position must contain exactly 6 elements (x, y, z, rx, ry, rz).")

    for _ in range(num_positions):
        new_position = [
            base_position[i] + random.uniform(-translation_range[i], translation_range[i])
            for i in range(3)
        ] + [
            base_position[i + 3] + random.uniform(-rotation_range[i], rotation_range[i])
            for i in range(3)
        ]
        positions.append(new_position)
    positions.append(base_position)
    return np.array(positions)


def save_positions(positions, filename="positions.csv"):
    np.savetxt(filename, positions, delimiter=',', fmt='%.3f')


robot = RobotClient("ws://192.168.10.1:1880")
robot.connect()

# Example usage:
base_position = robot.get_geom(matrix=False)  # Base position of the robotic arm
num_positions = 47  # Number of positions to generate
translation_range = (25, 25, 25)  # Allowable translation changes in mm
rotation_range = (8, 8, 8)  # Allowable rotation changes in degrees

positions = generate_random_positions(base_position, num_positions, translation_range, rotation_range)
save_positions(positions)