import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Unnormalized quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        Normalized quaternion with shape (4,)
    """
    # notice that the quaternion we use in this homework is in (w,x,y,z) order 
    # i*i = j*j = k*k = -1, i*j = k, j*i = -k, j*k = i, k*j = -i, k*i = j, i*k = -j
    # q = w + x*i + y*j + z*k
    # |q| = sqrt(q*(-q)) = (w + x*i + y*j + z*k) * (w - x*i - y*j - z*k) = w^2 + x^2 + y^2 + z^2
    # if we view the q as a 4D vector, we find that the norm is just the same as the L2 norm of the vector 
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q_normalized = q / q_norm
    else: 
        q_normalized = q
    return q_normalized

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Return the conjugate of the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The conjugate of the quaternion with shape (4,)
    """
    w,x,y,z = q
    return np.array([w, -x, -y, -z])

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply the two quaternions.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    np.ndarray
        The multiplication result with shape (4,)
    """
    # output = q1 * q2 = (w1 + x1*i + y1*j + z1*k) * (w2 + x2*i + y2*j + z2*k)
    #        = w1*w2 -x1*x2 - y1*y2 - z1*z2 + v1 x v2 + w1*v2 + w2*v1
    #        = w1*w2 -v1*v2 + w1*v2 + w2*v1 + v1 x v2
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + w2*x1 + y1*z2 - y2*z1
    y = w1*y2 + w2*y1 + x2*z1 - x1*z2
    z = w1*z2 + w2*z1 + x1*y2 - x2*y1
    return np.array([w, x, y, z])

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Use quaternion to rotate a 3D vector.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)
    v: np.ndarray
        Vector with shape (3,)

    Returns
    -------
    np.ndarray
        The rotated vector with shape (3,)
    """
    # first convert the vector to quaternion 
    v_quat = np.concatenate(([0], v))
    # use the formula q * v_quat * q^{-1} to rotate the vector
    # compute the q^{-1} = quat_conj(q) / |q|^2
    # notice that this homework requires the input quaternion to be normalized, so we can just use q_conj as the q^{-1}
    q_conj = quat_conjugate(q)

    q_rotate = quat_multiply(quat_multiply(q, v_quat), q_conj)

    return q_rotate[1:4]

def quat_relative_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute the relative rotation angle between the two quaternions.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    float
        The relative rotation angle in radians, greater than or equal to 0.
    """
    # 对于相对旋转，是否有顺序要求？经过简单计算，角度一样，但是转轴可能不同。
    q_relative = quat_multiply(q1, quat_conjugate(q2))
    q_relative = quat_normalize(q_relative)
    w = q_relative[0]

    # 由于数值误差，w可能会略微超过1或者-1，所以需要进行裁剪
    w = np.clip(w, -1.0, 1.0)
    angle = 2 * np.arccos(w) # already radians
    # 为什么要限制在pi呢？原因是如果算出来的角度大于pi，那么我们可以选择2pi-angle作为相对旋转的角度
    # 同时转轴还是同一个(方向相反），岂不是更好。
    if angle > np.pi:
        angle = 2 * np.pi - angle 
    return angle 


def interpolate_quat(q1: np.ndarray, q2: np.ndarray, ratio: float) -> np.ndarray:
    """
    Interpolate between two quaternions with given ratio.

    Please use Spherical linear interpolation (SLERP) here.

    When the ratio is 0, return q1; when the ratio is 1, return q2.

    The interpolation should be done in the shortest minor arc connecting the quaternions on the unit sphere.

    If there are multiple correct answers, you can output any of them.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)
    ratio: float
        The ratio of interpolation, should be in [0, 1]

    Returns
    -------
    np.ndarray
        The interpolated quaternion with shape (4,)

    Note
    ----
    What should be done if the inner product of the quaternions is negative?
    """
    # realize SLERP 
    # first compute the angle between the two quaternions
    # q1 and q2 is all normalized
    dot_product = np.dot(q1, q2) # the same as a,b vector inner product
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) 
    if angle > np.pi / 2:
        # we can use the opposite of q2 to get a smaller angle
        # why ? because q and -q represent the same rotation 
        # so compared with the angle between q1 and q2 , the angle between q1 and -q2 is smaller 
        q2 = -q2 
        angle = np.pi - angle 

    angle1 = angle * ratio 
    angle2 = angle * (1 - ratio)

    q_interpolated = (np.sin(angle1) * q2 + np.sin(angle2) * q1) / np.sin(angle)

    return q_interpolated



def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to rotation matrix.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    # hyw, just copy the formula from ppt
    w,x,y,z = q
    R00 = 1 - 2*y**2 - 2*z**2
    R01 = 2*x*y - 2*z*w
    R02 = 2*x*z + 2*y*w
    R10 = 2*x*y + 2*z*w
    R11 = 1 - 2*x**2 - 2*z**2 
    R12 = 2*y*z - 2*x*w
    R20 = 2*x*z - 2*y*w 
    R21 = 2*y*z + 2*x*w 
    R22 = 1 - 2*x**2 - 2*y**2
    R = np.array([[R00, R01, R02],
                  [R10, R11, R12],
                  [R20, R21, R22]])
    return R


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to quaternion.

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """
    # hyw, just copy the formula from ppt 
    R_trace = mat[0,0] + mat[1,1] + mat[2,2]
    w = np.sqrt(R_trace + 1) / 2
    x = (mat[2,1] - mat[1,2]) / (4*w)
    y = (mat[0,2] - mat[2,0]) / (4*w)
    z = (mat[1,0] - mat[0,1]) / (4*w)
    return np.array([w,x,y,z])


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi.

    If there are multiple answers, you can output any.

    Parameters
    ----------
    q: np.ndarray
        The quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """
    # convert the quaternion to rotation matrix first, 
    # then convert the rotation matrix to axis-angle representation 
    # R = quat_to_mat(q)
    # angle = np.arccos((R[0,0] + R[1,1] + R[2,2] - 1) / 2)
    # W = (R - R.T) / (2 * np.sin(angle))
    # axis = np.array([W[2,1], W[0,2], W[1,0]])
    # axis_angle = axis * angle 
    # return axis_angle 

    """
    另一种写法，直接从q本身推导，似乎更加简明
    """
    w,x,y,z = q
    half_angle = np.arccos(w)
    angle = 2 * half_angle
    w = np.array([x,y,z])
    if angle > np.pi:
        angle = 2 * np.pi - angle
        w = -w 
    angle_sin = np.sin(half_angle)
    if angle_sin == 0:
        axis_angle = np.zeros(3)
    else:
        axis = w / angle_sin
        axis_angle = axis * angle
    return axis_angle
    


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to quaternion.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """
    # so easy 
    angle = np.linalg.norm(aa)
    if angle == 0:
        return np.array([1,0,0,0])
    w = aa / angle 
    # here we can also limit the angle to be less than pi, and let w to be -w to get the same rotation result 
    # 这样一个是q，一个是-q，但是考虑到这个函数只是要求得到一个转化后的q，所以不用限制？
    # if angle > np.pi:
    #     angle = 2 * np.pi - angle
    #     w = -w 
    half_angle = angle / 2 
    quat = np.concatenate(([np.cos(half_angle)], w * np.sin(half_angle)))
    return quat

    


def axis_angle_to_mat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to rotation matrix.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    # we can use the function that we have already implemented to convert twice 
    quat = axis_angle_to_quat(aa)
    mat = quat_to_mat(quat)
    return mat 


def mat_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """
    quat = mat_to_quat(mat)
    axis_angle = quat_to_axis_angle(quat)
    return axis_angle 


def uniform_random_quat() -> np.ndarray:
    """
    Generate a random quaternion with uniform distribution.

    Returns
    -------
    np.ndarray
        The random quaternion with shape (4,)
    """
    # use uniform distribution to generate 
    # min,max ?
    quat = np.random.uniform(-1, 1, 4) # -1 is necessary, because we don't want to generate only positive quaternions 
    quat = quat_normalize(quat)
    return quat


def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    """
    Convert roll-pitch-yaw euler angles into rotation matrix.

    This is required since URDF use this as rotation representation.

    Parameters
    ----------
    rpy: np.ndarray
        The euler angles with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    roll, pitch, yaw = rpy

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = R_z @ R_y @ R_x  # Matrix multiplication in ZYX order
    return R
