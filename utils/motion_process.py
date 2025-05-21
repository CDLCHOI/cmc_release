import torch
from utils.quaternion import quaternion_to_cont6d, qrot, qinv
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from data_loaders.humanml.common.quaternion import *

example_data = np.load('dataset/HumanML3D/new_joints/000021.npy').reshape(-1, 22, 3)
tgt_skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(example_data[0]))

def recover_from_smpl(motion):
    joints = recover_from_rot(torch.from_numpy(motion), 22, tgt_skel).numpy()
    return joints

def global_to_local_xzy(r_pos, r_rot_quat):
    '''
    r_pos: (b,196,3) x y z  初始的x z一定是0, 即将人放在y轴上, 处于xz的原点
    data: (b,196,3)  x z y
    '''
    data = torch.zeros_like(r_pos).to(r_pos.device)
    # r_pos逐差计算相邻的xz差值
    data[:, :-1, :] = r_pos[:, 1:, :] - r_pos[:, 0:-1, :] # 只有195个
    data[:, -1, :] = data[:, -2, :] # 最后一个直接复制
    data_ = qrot(qinv(r_rot_quat), data) # x y z
    data[..., 0] = data_[..., 0]
    data[..., 1] = data_[..., 2]
    data[..., 2] = r_pos[..., 1] # 每个时刻根节点的y值即高度
    return data


def recover_root_rot_pos(data):
    rot_vel = data[..., 0] # (b,196) 根节点旋转角速度,第i个角速度表示第i帧向第i+1帧转的角速度
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1] # 所以这里r_rot_ang的第0帧是第0帧的真实朝向，为0，即已经向z轴了
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1) # 通过累加，得到的是每个时刻根节点相对于初始根节点即z轴的旋转角度

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device) # 获得每个时刻根节点的旋转的四元数
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3] # 每个时刻根节点的xz速度，即相邻的xz差值
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos) # 乘上对应的旋转，得到的是真正的相邻根节点的xz差值

    r_pos = torch.cumsum(r_pos, dim=-2) # 通过累加得到根节点的绝对位置

    r_pos[..., 1] = data[..., 3] # 每个时刻根节点的y值即高度

    # data_ = global_to_local_xzy(r_pos, r_rot_quat)
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    '''
    data: (L,263) or (B,L,263)
    '''
    r_rot_quat, r_pos = recover_root_rot_pos(data) # 这个支持batch操作
    if len(data.shape) == 3: # (B,L,22,3)
        data = data.flatten(0,1)
        r_rot_quat = r_rot_quat.flatten(0,1)
        r_pos = r_pos.flatten(0,1)
    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)
    ''' FK不支持batch操作，即
    cont6d_params: (L,22,6)
    r_pos: (L,22,3)
    '''
    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    assert type(data) == torch.Tensor
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4] # 这22*3个就是 ric
    positions = positions.view(positions.shape[:-1] + (-1, 3)) # (196,21,3)

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions
    

def get_cont6d_params(positions):
    face_joint_indx = [2, 1, 17, 16]
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")

    # (seq_len, joints_num, 4)
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

    '''Quaternion to continuous 6D'''
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    # (seq_len, 4)
    r_rot = quat_params[:, 0].copy() # 根节点的绝对旋转
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    velocity = qrot_np(r_rot[1:], velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1])) # 根节点的相对旋转
    # (seq_len, joints_num, 4)
    return cont_6d_params, r_velocity, velocity, r_rot

def foot_detect(positions, thres):
    fid_r, fid_l = [8, 11], [7, 10]
    velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    #     feet_l_h = positions[:-1,fid_l,1]
    #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    #     feet_r_h = positions[:-1,fid_r,1]
    #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
    return feet_l, feet_r

# def calc_redundant(ric_data):
#     '''
#     ric_data: (b,196,67)
#     '''
#     feet_thre = 0.002
#     positions = recover_from_ric(ric_data, 22)
#     cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
#     rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

#     local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
#                         global_positions[1:] - global_positions[:-1])
#     local_vel = local_vel.reshape(len(local_vel), -1)

#     feet_l, feet_r = foot_detect(positions, feet_thre)

#     data = ric_data[]
#     data = np.concatenate([data, ric_data[:-1]], axis=-1)
#     data = np.concatenate([data, rot_data[:-1]], axis=-1)
#     #     print(data.shape, local_vel.shape)
#     data = np.concatenate([data, local_vel], axis=-1)
#     data = np.concatenate([data, feet_l, feet_r], axis=-1)
#     return data

