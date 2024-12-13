from scipy.spatial.transform import Rotation as R
import numpy as np
# Define the Euler angles in degrees
euler_angles_deg = [45.60127175081633, 0.9272345695631321, 21.927734034132644]
# Convert to radians
euler_angles_rad = np.radians(euler_angles_deg)
# Create a rotation object from Euler angles
rotation = R.from_euler('xyz', euler_angles_rad)
# Invert the rotation
rotation_inv = rotation.inv()
# Convert the inverse rotation back to Euler angles in degrees
inv_euler_angles_deg = np.degrees(rotation_inv.as_euler('xyz'))
print(inv_euler_angles_deg)
