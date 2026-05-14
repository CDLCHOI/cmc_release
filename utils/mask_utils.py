import torch
import torch.nn.functional as F
import numpy as np
from utils.motion_process import recover_from_ric, recover_root_rot_pos, recover_from_rot
from utils.vis_utils import visualize_2motions, animate3d
from einops import rearrange
import matplotlib.pyplot as plt
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import cont6d_to_quat
import os

humanml_mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy'))[None, None, ...].cuda().float() # (1,1,263) dataset/HumanML3D/Mean.npy
humanml_std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy'))[None, None, ...].cuda()   # (1,1,263)
kit_mean = torch.from_numpy(np.load('dataset/KIT-ML/Mean.npy'))[None, None, ...].cuda().float()
kit_std = torch.from_numpy(np.load('dataset/KIT-ML/Std.npy'))[None, None, ...].cuda().float()

humanml_raw_mean = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Mean_raw.npy')).cuda()[None, None, ...].view(1,1,22,3) 
humanml_raw_std = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Std_raw.npy')).cuda()[None, None, ...].view(1,1,22,3)

kit_bone = [[0, 11], [11, 12], [12, 13], [13, 14], [14, 15], [0, 16], [16, 17], [17, 18], [18, 19], [19, 20], [0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [5, 6], [6, 7], [3, 8], [8, 9], [9, 10]]
t2m_bone = [[0,2], [2,5],[5,8],[8,11],
            [0,1],[1,4],[4,7],[7,10],
            [0,3],[3,6],[6,9],[9,12],[12,15],
            [9,14],[14,17],[17,19],[19,21],
            [9,13],[13,16],[16,18],[18,20]]
kit_kit_bone = kit_bone + (np.array(kit_bone)+21).tolist()
t2m_t2m_bone = t2m_bone + (np.array(t2m_bone)+22).tolist()

def random_window_mask(motion, real_length):
    '''
    为了加入时间上的补全功能。随机选取窗口比例0~1, 根据实际动作长度计算窗口长度, 随机选取窗口起始点idx, 将窗口部分的动作mask掉
    这里不会对motion的263向量进行随机mask
    '''
    B, L, num_features = motion.shape
    window_prob = torch.zeros((B,)).uniform_(0.2, 0.5).to(real_length.device) # 窗口比例0.2~0.5
    window_size = window_prob * real_length # 窗口长度
    low_bound = torch.rand((B,)).to(real_length.device) * (real_length - window_size) # 窗口下界
    high_bound = low_bound + window_size # 窗口上界
    batch_randperm = torch.arange(0,L).repeat(B,1).to(real_length.device)
    # 生成mask
    completion_mask = ~(torch.logical_and(batch_randperm >= low_bound[..., None], batch_randperm <= high_bound[..., None]))
    masked_motion = motion * completion_mask[..., None]
    return masked_motion


def calc_grad_scale(mask_hint):
    assert mask_hint.shape[1] == 196
    num_keyframes = mask_hint.sum(dim=1).squeeze(-1)
    max_keyframes = num_keyframes.max(dim=1)[0]
    scale = 20 / max_keyframes
    return scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


def root_dist_loss(gt_root, pred_root, real_mask, args):
    scale = 1 if args.dataset_name == 't2m' else 0.001
    gt_rot, gt_pos = recover_root_rot_pos(gt_root)
    pred_rot, pred_pos = recover_root_rot_pos(pred_root)
    if args.loss_type == 'l1':
        loss_rotate = F.l1_loss(gt_rot[real_mask], pred_rot[real_mask], reduction='mean')
        loss_position = F.l1_loss(gt_pos[real_mask] * scale, pred_pos[real_mask] * scale, reduction='mean')
    elif args.loss_type == 'l2':
        loss_rotate = F.mse_loss(gt_rot[real_mask], pred_rot[real_mask], reduction='mean')
        loss_position = F.mse_loss(gt_pos[real_mask] * scale, pred_pos[real_mask] * scale, reduction='mean')
    return loss_rotate, loss_position, gt_pos, pred_pos

def recover(motion):
    xyz = recover_from_ric(motion * humanml_std + humanml_mean, joints_num=22)
    return xyz

def vis_motion(pred_motion, gt_motion=None, dataset='t2m', save_path='./output/testsample/1.html', length=None, joints2_from_rot=False):
    """vis function for visualization conveniently

    Args:
        pred_motion (torch.Tensor): (L, dim)
        gt_motion (torch.Tensor): (L, dim)
        dataset (str, optional):
    """
    assert len(pred_motion.shape) == 2, f"got pred.shape = {pred_motion.shape}"
    if gt_motion is not None:
        assert len(gt_motion.shape) == 2, f"got gt.shape = {gt_motion.shape}"
        assert pred_motion.shape == gt_motion.shape, f"got pred.shape = {pred_motion.shape},  gt.shape = {gt_motion.shape}"
    dim = pred_motion.shape[-1]
    
    if dataset == 't2m':
        mean = np.load('dataset/HumanML3D/Mean.npy')[None, :dim] # dataset/HumanML3D/Mean.npy
        std = np.load('dataset/HumanML3D/Std.npy')[None, :dim]
        first_total_standard = 63
        bone_link = t2m_bone
        if gt_motion is not None:
            bone_link = t2m_t2m_bone
        joints_num = 22
        scale = 1#/1000
    else:
        mean = np.load('dataset/KIT/Mean.npy')[None, :dim] # dataset/HumanML3D/Mean.npy
        std = np.load('dataset/KIT/Std.npy')[None, :dim]
        first_total_standard = 60
        bone_link = kit_bone
        if gt_motion is not None:
            bone_link = kit_kit_bone
        joints_num = 21
        scale = 1/1000

    if type(pred_motion) == torch.Tensor:
        pred_motion = pred_motion.detach().cpu().numpy()
    if type(gt_motion) == torch.Tensor:
        gt_motion = gt_motion.detach().cpu().numpy()

    pred_motion = pred_motion * std + mean

    # convert to global position
    joint1 = recover_from_ric(torch.from_numpy(pred_motion).float(), joints_num).numpy()
    joint2 = None
    if gt_motion is not None:
        gt_motion = gt_motion * std + mean
        joint2 = recover_from_ric(torch.from_numpy(gt_motion).float(), joints_num).numpy()
        if joints2_from_rot:
            tgt_skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
            tgt_skel.get_offsets_joints(torch.from_numpy(joint1[0]))
            joint2 = recover_from_rot(torch.from_numpy(pred_motion), joints_num, tgt_skel).numpy()
        joint_original_forward = np.concatenate((joint1, joint2), axis=1)
    else:
        joint_original_forward = joint1

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    animate3d(joint_original_forward[:length]*scale, 
              BONE_LINK=bone_link, 
              first_total_standard=first_total_standard, 
              save_path=save_path) # 'init.html'
    return joint1, joint2

    # print(f'save motion html in {save_path}')
    # visualize_2motions(pred, std[..., :dim], mean[..., :dim], 't2m', None, motion2=None if gt is None else gt, save_path=save_path)

def complete_mask(traj_mask_263, traj_mask):
    control_id = traj_mask[0].sum(0).sum(-1).nonzero()
    traj_mask_263[..., :4] = True
    traj_mask_263[..., 4+3*(control_id-1):4+3*control_id] = True # ric  21*3
    return traj_mask_263

def calc_loss_xyz(pred, traj, traj_mask, dataset_name='t2m', only_err=False):
    assert type(pred) == torch.Tensor

    dim = pred.shape[-1]

    if dataset_name == 't2m':
        mean = humanml_mean[..., :dim]
        std = humanml_std[..., :dim]
        njoints = 22
    elif dataset_name == 'kit':
        mean = kit_mean[..., :dim]
        std = kit_std[..., :dim]
        njoints = 21
    else:
        raise NotImplementedError(f'{dataset_name} not supported')

    recon_xyz = recover_from_ric(pred * std + mean, joints_num=njoints)  # 反归一化再转全局xyz
    loss = F.l1_loss(recon_xyz[traj_mask], traj[traj_mask])
    mask_hint = traj.view(traj.shape[0], traj.shape[1], njoints, 3).sum(dim=-1, keepdim=True) != 0  
    # err = control_l2(recon_xyz.cpu().numpy(), traj.cpu().numpy(), mask_hint.cpu().numpy()).sum() / mask_hint.sum() 
    err = np.linalg.norm((recon_xyz.detach().cpu().numpy() - traj.detach().cpu().numpy()) * mask_hint.cpu().numpy(), axis=-1).sum() / mask_hint.sum() 
    if only_err:
        return err.item()
    else:
        return loss, err.item()
    
def calc_loss_xyz_perbatch(pred, traj, dataset_name='t2m'):
    '''
    计算一个batch中，每个batch数据自己的平均控制误差, 和最大误差
    '''
    assert type(pred) == torch.Tensor

    dim = pred.shape[-1]

    if dataset_name == 't2m':
        mean = humanml_mean[..., :dim]
        std = humanml_std[..., :dim]
        njoints = 22
    elif dataset_name == 'kit':
        mean = kit_mean[..., :dim]
        std = kit_std[..., :dim]
        njoints = 21
    else:
        raise NotImplementedError(f'{dataset_name} not supported')

    b = pred.shape[0]
    recon_xyz = recover_from_ric(pred * std + mean, joints_num=njoints)  # 反归一化再转全局xyz
    mask_hint = traj.view(traj.shape[0], traj.shape[1], njoints, 3).sum(dim=-1, keepdim=True) != 0  # (b,196,22,1)
    control_joint_num = len(set(mask_hint.nonzero()[:,2].tolist()))
    assert control_joint_num == 1, "only support control 1 joint !!!"
    control_joint_num_and_frames = np.array([len(mask_hint[i].nonzero()) for i in range(b)])   # 每个batch数据中实际控制的 帧数*关节数
    err = np.linalg.norm((recon_xyz.detach().cpu().numpy() - traj.detach().cpu().numpy()) * mask_hint.cpu().numpy(), axis=-1) # (32,196,22)
    err_temporal = err.sum(-1) / control_joint_num  # (32,196)
    err_per_max = err_temporal.max(-1)
    err_per_mean = err.sum((1,2)) / control_joint_num_and_frames
    return err_per_mean, err_per_max

def calc_err_perframe(pred, traj, traj_mask, dataset_name='t2m', joint_id=0):
    assert type(pred) == torch.Tensor

    dim = pred.shape[-1]

    if dataset_name == 't2m':
        mean = humanml_mean[..., :dim]
        std = humanml_std[..., :dim]
    elif dataset_name == 't2m':
        mean = kit_mean[..., :dim]
        std = kit_std[..., :dim]
    else:
        raise NotImplementedError(f'{dataset_name} not supported')

    recon_xyz = recover_from_ric(pred * std + mean, joints_num=22)  # 反归一化再转全局xyz
    loss = F.l1_loss(recon_xyz[traj_mask], traj[traj_mask])
    mask_hint = traj.view(traj.shape[0], traj.shape[1], 22, 3).sum(dim=-1, keepdim=True) != 0  
    # err = control_l2(recon_xyz.cpu().numpy(), traj.cpu().numpy(), mask_hint.cpu().numpy()).sum() / mask_hint.sum() 
    err = np.linalg.norm((recon_xyz[:,:,joint_id,:].cpu().numpy() - traj[:,:,joint_id,:].cpu().numpy()) * mask_hint[:,:,joint_id,:].cpu().numpy(), axis=-1)
    return loss, err


def load_ckpt(net, path):
    ckpt = torch.load(path, map_location='cpu')
    if 'module' in list(ckpt['trans'].keys())[0]:
        new_ckpt = {}
        for k, v in ckpt['trans'].items():
            new_k = k.replace('module.', '') if 'module' in k else k
            new_ckpt[new_k] = v
        net.load_state_dict(new_ckpt, strict=True)
    else:
        net.load_state_dict(ckpt['trans'], strict=True)

def draw_xz_map(gt, pred, out_dir='./output/testsample/'):
    x = gt[0,:,0].detach().cpu().numpy()
    z = gt[0,:,2].detach().cpu().numpy()
    plt.scatter(x, z, c='r')
    x = pred[0,:,0].detach().cpu().numpy()
    z = pred[0,:,2].detach().cpu().numpy()
    plt.scatter(x[::10], z[::10])
    plt.savefig(f'{out_dir}/xz.png')

def generate_src_mask(T, length):
    B = len(length)
    mask = torch.arange(T).repeat(B, 1).to(length.device) < length.unsqueeze(-1)
    return mask

def load_ckpt(net, ckpt_path, key=None, strict=True):
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if key:
        ckpt = ckpt[key]

    # 去掉module
    if 'module' in list(ckpt.keys())[0]:
        new_ckpt = {}
        for k, v in ckpt.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            new_ckpt[new_k] = v

    model_dict = net.state_dict()
    ckpt = {k: v for k, v in ckpt.items() if k in model_dict} # 过滤不存在的key
    ckpt = {k: v for k, v in ckpt.items() if model_dict[k].shape==v.shape} # 过滤存在但形状不一致的key
    assert ckpt != {}, 'ckpt is EMPTY'
    model_dict.update(ckpt)
    _, unexpect_keys = net.load_state_dict(model_dict, strict=strict)
    print('unexpect_keys=',unexpect_keys)
    

##### ---- CLIP ---- #####
# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb
    

