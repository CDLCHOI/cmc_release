import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

from torch.utils.data._utils.collate import default_collate
from utils.word_vectorizer import WordVectorizer
from utils.motion_process import recover_root_rot_pos, recover_from_ric, recover_from_rot
import sys

from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import cont6d_to_quat
from data_loaders.humanml.utils.get_opt import get_opt

def create_trajmask263(joint_ids, frames=None, dataset_name='t2m', mode='train'):
    """ create trajectory mask for motion representation in HumanML3D/KIT for DiffMoAE

    Args:
        joint_ids (np.ndarray): 
        frames (np.ndarray):
    Returns:
        traj_mask: (L, 22, 3)    for calculating global xyz loss
        traj_mask_263: (L, 263)  for DiffMoAE
    """
    assert isinstance(joint_ids, np.ndarray)
    if frames is None:
        frames = np.arange(L)
    else:
        assert isinstance(frames, np.ndarray)

    L = 196

    if dataset_name == 't2m':
        traj_mask = np.zeros((L, 22, 3)).astype(bool)
        traj_mask_263 = np.zeros((L, 263)).astype(bool)
    elif dataset_name == 'kit':
        traj_mask = np.zeros((L, 21, 3)).astype(bool)
        traj_mask_263 = np.zeros((L, 251)).astype(bool)
    else:
        raise NotImplementedError(f'{dataset_name} not supported')

    traj_mask_263[:, :4] = True # root
    for i in joint_ids:
        traj_mask[frames, i] = True
        traj_mask_263[frames, 4+3*(i-1):4+3*i] = True # ric  21*3

    return traj_mask, traj_mask_263

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text motion matching model, and evaluations'''
class ControlDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, mode, 
                 control_joint=0, density=100, dataset_name='t2m', 
                 multi_joint_control=False, unit_length=None):
        
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.split_file = split_file
        self.mode = mode
        self.control_joint = control_joint
        self.density = density
        self.multi_joint_control = multi_joint_control
        # print('self.control_joint = ', self.control_joint)
        # print('self.density = ', self.density)
        assert  0 <= self.density <= 100, "density should be in [0, 100], got {}".format(self.density)

        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        

        fps = 20 if self.opt.dataset_name == 't2m' else 12.5 # HumanML3D帧率20; KIT帧率12.5

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        
        # id_list = id_list[:66]; print('id_list[32:64]for debug !!!!!!!! ')
        # id_list = id_list[:222] if sys.gettrace() else id_list
        # if self.mode == 'debug':
        # id_list = id_list[:32]
        
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                # 这部分的意思是，一段动作，但是有的文本对应的动作段是有起始点和结束点的
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag  # 起始秒 from_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag # 结束秒 to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * fps) : int(to_tag * fps)] # 起始秒和结束秒 乘上帧率 KIT帧率是12.5
                                if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200): # 过滤长度过短过长的动作
                                    continue
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))

                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text': text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1])) # 把名字长度二元组依长度升序排列
        name_list = new_name_list # debug用
        # length_list = length_list # debug用

        self.mean = mean
        self.std = std
        if 'HumanML3D' in opt.data_root:
            spatial_norm_path = './dataset/humanml_spatial_norm'
            n_joints = 22
        elif 'KIT' in opt.data_root:
            spatial_norm_path = './dataset/kit_spatial_norm'
            n_joints = 21
        else:
            raise NotImplementedError('unknown dataset')
        # 全局xyz的均值和方差；原本HumanML3D的Mean.npy是相对xyz的均值和方差
        self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy')).reshape(n_joints,3)
        self.raw_std = np.load(pjoin(spatial_norm_path, 'Std_raw.npy')).reshape(n_joints,3)
        
        self.data_dict = data_dict
        self.name_list = name_list
        print(f'=== total {len(self.data_dict)} data')
        
    
    def transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * std + mean
    

    def random_mask_train(self, joints, n_joints=22):
        if n_joints == 22:
            controllable_joints = np.array([0, 10, 11, 15, 20, 21])
            joints_name = np.array(['pelvis', 'left_foot', 'right_foot', 'head', 'left_wrist', 'right_wrist'])
        elif n_joints == 21:
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF'] # 根，头，左手，右手，左脚，右脚
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])
        else:
            raise NotImplementedError
        
        if isinstance(self.control_joint, list):
            if self.control_joint == [-1]: # default -1, 随机选取控制关节数
                num_joints = len(controllable_joints)
                if self.multi_joint_control:
                    num_joints_control = np.random.choice(np.arange(1, num_joints+1), 1) # 1~6  多关节控制
                else:
                    num_joints_control = 1
                choose_joint = np.random.choice(num_joints, num_joints_control, replace=False) # 选择控制的关节点
                choose_joint = controllable_joints[choose_joint]
            else:
                choose_joint = np.array(self.control_joint)
        else:
            pass

        # print(choose_joint)
        assert set(choose_joint).issubset(controllable_joints), choose_joint



        # 选择控制帧比例
        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1 # 随机设定控制的帧数 范围 [1,L-1]
        if self.density:
            if self.density in [1, 2, 5]:
                choose_seq_num = self.density
            else:
                choose_seq_num = int(length * self.density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False) # 根据帧数选择控制的时刻帧
        choose_seq.sort()

        traj_mask, traj_mask_263 = create_trajmask263(choose_joint, choose_seq, dataset_name=self.opt.dataset_name, mode=self.mode)

        # normalize
        joints = (joints - self.raw_mean) / self.raw_std
        joints = joints * traj_mask[:length, ...]
        return joints, traj_mask_263, traj_mask
    


    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        '''随机性
        1. 文本随机 text_data = random.choice(text_list)
        2. coin2 = np.random.choice(['single', 'single', 'double'])
        3. 动作随机起点截取 idx = random.randint(0, len(motion) - m_length)
        '''
        idx = self.pointer + item
        # idx = 29
        # idx = 1316
        # idx = 29; print(f' idx={idx} , {self.name_list[idx]} for debug')
        # idx = 120; print(f' idx={idx} for debug') # 站着挥右手
        # idx = 179; print(f' idx={idx}for debug') # 站着挥双手,错误人体，正反面反过来的
        filename = self.name_list[idx]
        # filename = '009613'; print(f' filename = {filename} for debug')
        # filename = '004822'; print(f' filename = {filename} for debug') # example1
        # filename = '002662'; print(f' filename = {filename} for debug')
        # filename = '008382'; print(f' filename = {filename} for debug')
        data = self.data_dict[filename]

        # data = self.data_dict['000007'] 
        # if idx <= 32:
        
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # motion = np.load(f'dataset/HumanML3D/new_joint_vecs/{self.name_list[idx]}.npy'); print('for debug !!!')
        # m_length = 199; print('for debug !!!')

        # Randomly select a caption
        # if self.mode == 'train':
        text_data = random.choice(text_list)
        # else:
        # text_data = text_list[0]; print('choose 0th text, for debug !!!') # ① 固定文本
        caption, tokens = text_data['caption'], text_data['tokens']

        

        if len(tokens) < self.opt.max_text_len:
            # 句子短，补SOS和EOS token，然后补unknown token至固定长度
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # 句子场，固定切割到固定长度，再补SOS EOS
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)        # (22,15)   22是预设的句子最大长度20加上2个SOS和EOS
        word_embeddings = np.concatenate(word_embeddings, axis=0)  # (22,300)

        # 将动作长度截取为unit_length即4的整数倍，并通过coin2引入一些随机，无需细抠
        # if self.mode == 'train':
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        # coin2 = 'single'; print('for debug !!!') # ② 固定coin
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        # else:
        #     m_length = (m_length // self.opt.unit_length) * self.opt.unit_length

        # if self.mode == 'train':
        idx = random.randint(0, len(motion) - m_length)
        # else:
        # idx = 0; print('for debug !!!') # ③ 固定初始帧

        motion = motion[idx:idx+m_length]

        n_joints = 22 if motion.shape[-1] == 263 else 21
        # hint is global position of the controllable joints
        joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints) # (L, 22, 3)  每个关节点的全局坐标
        joints = joints.numpy()
        ##########################
        # joints_disk = np.load(f'dataset/HumanML3D/new_joints/{self.name_list[idx]}.npy')
        # assert np.allclose(joints, joints_disk[:196])

        # tgt_skel = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, 'cpu')
        # tgt_skel.get_offsets_joints(torch.from_numpy(joints[0]))
        # joints_rot1 = recover_from_rot(torch.from_numpy(motion), n_joints, tgt_skel).numpy()
        # assert np.allclose(joints, joints_rot1, atol=1e-6)

        # example_data = np.load('/data/motion/HumanML3D/new_joints/000021.npy')
        # example_data = example_data.reshape(len(example_data), -1, 3)
        # example_data = torch.from_numpy(example_data)
        # tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
        # joints_rot2 = recover_from_rot(torch.from_numpy(motion), n_joints, tgt_skel).numpy()
        # assert np.allclose(joints, joints_rot2, atol=1e-6)
        ##########################

        # control any joints at any time
        hint, traj_mask_263, traj_mask = self.random_mask_train(joints, n_joints) # joints: (L,22,3)
        hint = hint.reshape(hint.shape[0], -1) # (L,22*3)

        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            hint   = np.concatenate([hint, np.zeros((self.max_motion_length - m_length, hint.shape[1])) ], axis=0)
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, motion.shape[1])) ], axis=0)

        hint = hint.astype(np.float32).reshape(self.max_motion_length, n_joints, 3)
        motion = motion.astype(np.float32)
        

        # 确保取得的轨迹以及traj_mask正确
        # joints: L,22,3
        # hint: 196,22,3
        assert np.allclose(joints * traj_mask[:m_length, ...] , ((hint * self.raw_std + self.raw_mean) * traj_mask)[:m_length, ...], atol=1e-4) # HumanML3D这里阈值可以是1e-6，kit只能是1e-4

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), hint, traj_mask_263, traj_mask, filename

# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train", control_joint=0, multi_joint_control=False, unit_length=None, density=100, **kwargs):
        self.mode = mode
        self.split = split
        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir) # ./dataset/HumanML3D/new_joint_vecs
        opt.text_dir = pjoin(abs_base_path, opt.text_dir) # ./dataset/HumanML3D/texts
        opt.data_root = pjoin(abs_base_path, opt.data_root) # ./dataset/HumanML3D/
        opt.meta_dir = './dataset'
        self.opt = opt
        self.dataset_name = opt.dataset_name
        print('Loading dataset %s ...' % opt.dataset_name)

        if mode == 'gt':
            # used by T2M models (including evaluators) 这就是MDM里的t2m_mean.npy 即 t2m/Comp_v6_KLD005/meta/mean.npy
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy')) # dataset/HumanML3D/Mean.npy
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt') # dataset/HumanML3D/train.txt

        self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
        self.t2m_dataset = ControlDataset(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, mode,  
                                          control_joint=control_joint, density=density, dataset_name=self.dataset_name, 
                                          multi_joint_control=multi_joint_control,
                                          unit_length=unit_length)
        self.num_actions = 1 # dummy placeholder


    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
# class KIT(HumanML3D):
#     def __init__(self, mode, datapath='./dataset/kit_opt.txt', split="train", **kwargs):
#         super(KIT, self).__init__(mode, datapath, split, **kwargs)



def DataLoader(batch_size, args, shuffle=False, mode='train', split='train', num_workers=8, drop_last=True) : 
    if args.dataset_name == 't2m':
        datapath = './dataset/humanml_opt.txt'
    elif args.dataset_name == 'kit':
        datapath = './dataset/kit_opt.txt'
    else:
        raise NotImplementedError
    
    dataset = HumanML3D(mode, datapath=datapath, split=split, control_joint=args.control_joint, 
                            density=args.density, multi_joint_control=args.multi_joint_control, unit_length=2*args.down_t,)

    if batch_size == 1:
        num_workers = 0
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, collate_fn=collate_fn if split=='test' else None, shuffle=shuffle, num_workers=num_workers, drop_last = drop_last)
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

if __name__ == '__main__':
    train_loader = DataLoader(batch_size=1, mode='train')
    train_loader_iter = cycle(train_loader)