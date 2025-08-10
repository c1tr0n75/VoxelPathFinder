import numpy as np
import os

# Path to a generated file (cross-platform, relative to this script)
fname = os.path.join(os.path.dirname(__file__), "sample_0620.npz")

# Load
data = np.load(fname)
print("Arrays in this file:", data.files)

# Show shapes/types
print("voxel_data shape:", data['voxel_data'].shape)      # (3, 32, 32, 32)
print("positions:", data['positions'])                    # [[start_x start_y start_z], [goal_x goal_y goal_z]]
print("target_actions:", data['target_actions'])
print("num_turns:", data['num_turns'])
print("path_length:", data['path_length'])

# Example: visualize a 2D obstacle slice (Z = 16)
import matplotlib.pyplot as plt
plt.imshow(data['voxel_data'][0, :, :, 16], cmap='gray')
plt.title('Obstacle map (XY slice at Z=16)')
plt.show()
