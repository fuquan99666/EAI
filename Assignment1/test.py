import numpy as np
from rotation import axis_angle_to_mat
mat = axis_angle_to_mat(np.array([0, 0, np.pi/2]))
print(mat)