
import torch
import numpy as np
from utils import rotation_conversions 
from utils.motion_process import recover_root_rot_pos

def normalize(v):
    """Normalize vectors in the last dimension."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norm

def rotmat_to_axis_angle(R):
    """
    Convert batch of rotation matrices to axis-angle representation.
    
    Input: R -> shape (..., 3, 3)
    Output: axis_angle -> shape (..., 3)
    """
    original_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    angle = np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))
    axis = np.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1]
    ], axis=-1)
    axis = normalize(axis)

    axis_angle = axis * angle[:, np.newaxis]
    axis_angle = axis_angle.reshape(original_shape + (3,))
    return axis_angle

def sixd_to_rotmat(x):
    """
    Convert 6D representation to rotation matrix.
    
    Input: x -> shape (..., 6)
    Output: R -> shape (..., 3, 3)
    """
    original_shape = x.shape[:-1]
    x = x.reshape(-1, 6)

    a1 = x[:, 0:3]
    a2 = x[:, 3:6]

    b1 = normalize(a1)
    dot_product = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = normalize(a2 - dot_product * b1)
    b3 = np.cross(b1, b2)

    R = np.stack((b1, b2, b3), axis=-1)
    R = R.reshape(original_shape + (3, 3))
    return R

def sixd_to_axis_angle(sixd):
    """
    Convert 6D rotation to axis-angle. Supports arbitrary leading dimensions.
    
    Input: sixd -> shape (..., 6)
    Output: axis_angle -> shape (..., 3)
    """
    # Convert 6D to rotation matrix
    rot_mats = sixd_to_rotmat(sixd)
    # Convert rotation matrix to axis-angle
    axis_angle = rotmat_to_axis_angle(rot_mats)
    return axis_angle

def convert_hml2smpl(motion):
    '''
    input：(l,263)x z y r 21*3 21*6 22*3 4
    return： (l,23,3)
    '''
    L, dim = motion.shape
    n_joints = 22 if dim == 263 else 21
    # root_xyz = motion[:,:3].transpose(0,2,1)

    rotations_6d = motion[:,67:193].reshape(-1,n_joints-1,6)
    rotations_mat = rotation_conversions.rotation_6d_to_matrix(torch.Tensor(rotations_6d))
    rotations_axis_angle = rotation_conversions.matrix_to_axis_angle(rotations_mat)

    root_quat, root_xyz = recover_root_rot_pos(torch.Tensor(motion))
    root_axis_angle = rotation_conversions.quaternion_to_axis_angle(root_quat)

    smpl = np.zeros((L, (n_joints+1)*3)) # 22*3的旋转，1*3的根xyz
    smpl[:,:3] = root_axis_angle  # 根节点的相对于初始帧的绝对旋转的轴角表示
    smpl[:,3:-3] = rotations_axis_angle.flatten(1,2) # 非根节点的轴角表示
    smpl[:,-3:] = root_xyz # 根节点的全局xyz
    return smpl

if __name__ == '__main__':
    hml = np.load('/home/deli/project/cmc_release/dataset/HumanML3D/new_joint_vecs/000000.npy')
    smpl = convert_hml2smpl(hml)


