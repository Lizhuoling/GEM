import numpy as np
import torch
import os
import random
import cv2
import h5py
import pdb
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def vis_attn():
    rgb_image = cv2.imread('cam_vis.png')
    attention_map = np.load('attn_vis.npy')
    max_attn = attention_map[2][7]
    exchange_attn = attention_map[6][9]
    attention_map[2][7] = exchange_attn
    attention_map[6][9] = max_attn

    resized_attention_map = cv2.resize(attention_map, (rgb_image.shape[1], rgb_image.shape[0]), interpolation = cv2.INTER_CUBIC)
    normalized_attention_map = resized_attention_map / np.max(resized_attention_map)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.imshow(resized_attention_map, cmap='Spectral_r', alpha=0.8)
    plt.axis('off')
    plt.savefig('overlay_image.png', bbox_inches='tight', pad_inches=0)
    
def check_track(track_path):
    for filename in sorted(os.listdir(track_path)):
        with h5py.File(os.path.join(track_path, filename), 'r') as f:
            a = f['exterior_image_1_left_pred_tracks'][:]
    print('Done!')
    
def vis_attn():
    rgb_image = cv2.imread('cam_vis.png')
    attention_map = np.load('attn_vis.npy')
    max_attn = attention_map[2][7]
    exchange_attn = attention_map[6][9]
    attention_map[2][7] = exchange_attn
    attention_map[6][9] = max_attn
    resized_attention_map = cv2.resize(attention_map, (rgb_image.shape[1], rgb_image.shape[0]), interpolation = cv2.INTER_CUBIC)
    normalized_attention_map = resized_attention_map / np.max(resized_attention_map)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.imshow(resized_attention_map, cmap='Spectral_r', alpha=0.8)
    plt.axis('off')
    plt.savefig('overlay_image.png', bbox_inches='tight', pad_inches=0)
    
def quaternion_to_rotation_matrix(q):
    """
    Description:
        Convert a (n, 4) quaternion to a (n, 3, 3) rotation matrix.
    Input:
        q: A batch of quaternions. shape: (n, 4)
    Output:
        R: A batch of rotation matrices. shape: (n, 3, 3)
    """
    q_x, q_y, q_z, q_w = q.unbind(dim=-1)  # shape: (n,)
    R = torch.stack([
        1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_w*q_z), 2*(q_x*q_z + q_w*q_y),
        2*(q_x*q_y + q_w*q_z), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_w*q_x),
        2*(q_x*q_z - q_w*q_y), 2*(q_y*q_z + q_w*q_x), 1 - 2*(q_x**2 + q_y**2)
    ], dim=-1).reshape(-1, 3, 3)  # shape: (n, 3, 3)
    return R

def quaternion_conjugate(q):
    """
    Description:
        Batch quaternion conjugate for xyzw order.
    """
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack([-x, -y, -z, w], dim=-1)

def quaternion_multiply(q1, q2):
    """
    Description:
        Batch quaternion multiplication for xyzw order.
    """
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([x, y, z, w], dim=-1)

def quaternion_angle_diff(A, B):
    '''
    Description:
        Compute the angle difference between two quaternions.
    '''
    dot_product = torch.sum(A * B, dim=1)
    angle_diff = 2 * torch.acos(dot_product)
    angle_diff = torch.min(angle_diff, 2 * torch.pi - angle_diff)
    return angle_diff

def get_quaternion_A2B(A, B):
    '''
    Description:
        Compute the quaternion that rotates vector A to vector B.
    Input:
        A shape: (n, 4)
        B shape: (n, 4)
    '''
    A_inv = quaternion_conjugate(A)
    delta_quat = quaternion_multiply(B, A_inv)
    return delta_quat
    
def quaternion_rotate(a, q):
    """
    Description:
        Batch quaternion rotation. Rotate object pose a by the quaternion angle q.
    """
    q_inv = quaternion_conjugate(q)
    b = quaternion_multiply(quaternion_multiply(q, a), q_inv)
    return b

def euler_zyx_to_quaternion(euler_angles, degrees=True):
    """
    Convert ZYX Euler angles (yaw, pitch, roll) to quaternions.
    
    Args:
        euler_angles (torch.Tensor): Tensor of shape (n, 3) representing ZYX Euler angles in the format [yaw, pitch, roll].
        degrees (bool, optional): If True, the input Euler angles are in degrees. If False, they are in radians. Default is True.
    
    Returns:
        torch.Tensor: Tensor of shape (n, 4) representing quaternions in the format [x, y, z, w].
    """
    # Convert to radians if input is in degrees
    if degrees:
        euler_angles = euler_angles * (torch.pi / 180.0)
    
    # Extract Euler angles
    yaw, pitch, roll = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    
    # Compute quaternion components
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    # Quaternion components
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    
    # Combine quaternion components into a single tensor
    quaternions = torch.stack([x, y, z, w], dim=1)
    
    return quaternions

def quaternion_to_euler_zyx(quaternions, degrees=True):
    """
    Convert quaternions to ZYX Euler angles (yaw, pitch, roll).
    
    Args:
        quaternions (torch.Tensor): Tensor of shape (n, 4) representing quaternions in the format [x, y, z, w].
        degrees (bool, optional): If True, the output Euler angles are in degrees. If False, they are in radians. Default is True.
    
    Returns:
        torch.Tensor: Tensor of shape (n, 3) representing Euler angles in the format [yaw, pitch, roll].
    """
    # Extract quaternion components
    x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Compute yaw (rotation around Z-axis)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # Compute pitch (rotation around Y-axis)
    pitch = torch.asin(2 * (w * y - z * x))
    
    # Compute roll (rotation around X-axis)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    
    # Combine Euler angles into a single tensor
    euler_angles = torch.stack([yaw, pitch, roll], dim=1)
    
    # Convert to degrees if required
    if degrees:
        euler_angles = euler_angles * (180.0 / torch.pi)
    
    return euler_angles

def normal_from_cross_product(points_2d: torch.Tensor) -> torch.Tensor:
    """
    Compute the normal vectors for 3D points in multiple images.

    Parameters:
    points_2d: torch.Tensor, shape (n, h, w, 3), representing the 3D coordinates of points in n images.

    Returns:
    xyz_normal: torch.Tensor, shape (n, h, w, 3), representing the normal vectors for each point.
    """
    # Pad the array
    xyz_points_pad = torch.nn.functional.pad(points_2d, (0, 0, 0, 1, 0, 1), mode="replicate")
    # Compute the vertical vector differences
    xyz_points_ver = xyz_points_pad[:, :-1, :-1, :] - xyz_points_pad[:, 1:, :-1, :]
    # Compute the horizontal vector differences
    xyz_points_hor = xyz_points_pad[:, :-1, :-1, :] - xyz_points_pad[:, :-1, 1:, :]
    # Compute the normal vectors using the cross product
    xyz_normal = torch.cross(xyz_points_hor, xyz_points_ver, dim=-1)
    # Compute the magnitudes of the normal vectors
    xyz_dist = torch.norm(xyz_normal, dim=-1, keepdim=True)
    # Normalize the normal vectors
    xyz_normal = torch.where(xyz_dist != 0, xyz_normal / xyz_dist, torch.zeros_like(xyz_normal))
    return xyz_normal

def numpy_euler_to_quaternion(euler_angles, degrees=True):
    """
    Convert Euler angles to a quaternion.

    Parameters:
        euler_angles (list or tuple): [yaw, pitch, roll] in degrees or radians.
        degrees (bool): If True, input angles are in degrees. Otherwise, they are in radians.

    Returns:
        quaternion (list): [x, y, z, w] quaternion.
    """
    # Create a Rotation object from Euler angles
    r = Rotation.from_euler('zyx', euler_angles, degrees=degrees)
    # Get the quaternion representation
    quaternion = r.as_quat()
    return quaternion
    
if __name__ == '__main__':
    check_track(track_path = '/home/cvte/twilight/home/data/droid_h5py_tiny/tracks')
    
    
