import os
import sys
import torch
import tqdm
import random
import numpy as np
import codecs as cs

from os.path import join as pjoin
from utils.word_vectorizer import WordVectorizer
from options.get_eval_option import get_opt
from utils import rotation_conversions 
from utils.motion_process import recover_root_rot_pos

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def DataLoader(args, diffusion,  split='train', shuffle=False, num_workers=8, drop_last=True) : 
    dataset = CriticDataset(args, diffusion, split=split, datatype=args.datatype)
    
    if args.batch_size == 1:
        num_workers = 0
    train_loader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=shuffle, num_workers=num_workers, drop_last = drop_last)
    
    return train_loader

class CriticDataset:
    def __init__(self, args, diffusion, datapath='./dataset/humanml_opt.txt', split='train', 
                 mode='train', dataset_name='t2m', datatype='smpl'):
        assert datatype in ['hml', 'smpl']
        self.args = args
        self.diffusion = diffusion
        self.noisy_step_list = np.linspace(0, diffusion.num_timesteps, args.num_noisy_timesteps+1)[1:]-1 # [99,199,...,999]

        # 获取数据集opt
        abs_base_path = f'.' 
        opt = get_opt(datapath)
        split_file = pjoin(opt.data_root, f'{split}.txt') # dataset/HumanML3D/train.txt
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir) # ./dataset/HumanML3D/new_joint_vecs
        opt.text_dir = pjoin(abs_base_path, opt.text_dir) # ./dataset/HumanML3D/texts
        opt.data_root = pjoin(abs_base_path, opt.data_root) # ./dataset/HumanML3D/
        opt.meta_dir = pjoin(abs_base_path, './dataset')
        self.opt = opt

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


        self.w_vectorizer = WordVectorizer('./glove', 'our_vab')
        self.max_motion_length = args.max_motion_length
        self.mode = mode
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        

        fps = 20 if self.opt.dataset_name == 't2m' else 12.5 # HumanML3D帧率20; KIT帧率12.5

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        name_list = []
        length_list = []
        
        # id_list = id_list[:66]; print('id_list[32:64]for debug !!!!!!!! ')
        # id_list = id_list[:88] if sys.gettrace() else id_list
        # if self.mode == 'debug':
        # id_list = id_list[:32]

        # 读取缓存数据更快
        opt.cache_dir = '/home/deli/.cache/huggingface/hub/models--guytevet--CLoSD/snapshots/de7106b947b6f70700b5320d1cd61fef4a9ebc9b'
        cache_path = pjoin(opt.cache_dir, 'data', 'humanml3d', self.opt.dataset_name + '_' + split + '.npy')
        if os.path.exists(cache_path):
            print(f'Loading motions from cache file [{cache_path}]...')
            _cache = np.load(cache_path, allow_pickle=True)[None][0]
            name_list, length_list, data_dict = _cache['name_list'], _cache['length_list'], _cache['data_dict']
        else:
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
                                    name_list.append(new_name)
                                    length_list.append(len(n_motion))

                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)

                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data}
                        name_list.append(name)
                        length_list.append(len(motion))
                except:
                    pass

        # name_list, length_list = zip(*sorted(zip(name_list, length_list), key=lambda x: x[1])) # 把名字长度二元组依长度升序排列
        # name_list = name_list # debug用
        # length_list = length_list # debug用
        
        self.data_dict = data_dict
        self.name_list = name_list
        print(f'=== total {len(self.data_dict)} data')

    def transform(self, data, mean=None, std=None):
        assert isinstance(data, np.ndarray)
        if mean is None and std is None:
            return (data - self.mean) / self.std
        else:
            return (data - mean) / std

    def inv_transform(self, data, mean=None, std=None):
        assert isinstance(data, np.ndarray)
        if mean is None and std is None:
            return data * self.std + self.mean
        else:
            return data * std + mean
        
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        filename1 = self.name_list[idx]
        # filename = '009613'; print(f' filename = {filename} for debug')
        data1 = self.data_dict[filename1]
        motion1, m_length1, text_list = data1['motion'], data1['length'], data1['text']

        # 50%概率选取同motion
        if np.random.choice([0,1]): 
            idx2 = idx
        else:
            idx2 = np.random.choice(self.__len__()) 
        filename2 = self.name_list[idx2]
        data2 = self.data_dict[filename2]
        motion2, m_length2, text_list = data2['motion'], data2['length'], data2['text'] 

        # 返回类型torch.Tensor
        m1 = self.preprocess_motion(motion1, m_length1) # (196,23*3) 轴角表示
        m2 = self.preprocess_motion(motion2, m_length2)
        

        # 挑选一个t进行加噪
        t1, t2 = self.get_noisy_t_v2()
        xt1 = self.diffusion.q_sample(m1, t1)
        xt2 = self.diffusion.q_sample(m2, t2)
        
        xt1 = xt1.float()
        xt2 = xt2.float()

        return m1, m2, xt1, xt2, m_length1, m_length2, t1, t2
    
    def get_noisy_t(self):
        ''' 纯随机选t，网络估计太好训练了 '''
        t = np.random.choice(self.noisy_step_list, size=2, replace=False).astype(int)
        t = np.sort(t)
        t = torch.tensor(t)
        assert t[0] < t[1]
        return t
    
    def get_noisy_t_v2(self):
        ''' 0-1000选一个点，随机一个区间长度50-100，再区间长度里再挑2个t '''
        # point = np.random.choice(self.diffusion.num_timesteps) # 应该用这个，下一行只是暂时实现方便
        point = np.random.choice(np.arange(100, 900))
        interval_length = np.random.choice(np.arange(20, 50))
        random_list = np.arange(point-interval_length//2, point+interval_length//2)
        t = np.random.choice(random_list, size=2, replace=False).astype(int)
        t = np.sort(t)
        t = torch.tensor(t)
        assert t[0] < t[1]
        return t



    def preprocess_motion(self, motion, m_length):
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

        # if self.mode == 'train':
        idx = random.randint(0, len(motion) - m_length)
        # else:
        # idx = 0; print('for debug !!!') # ③ 固定初始帧

        motion = motion[idx:idx+m_length]

        # motion 263的归一化
        motion = self.transform(motion)

        
        if self.args.datatype == 'smpl':
            motion_tensor = self.hml2smpl(torch.tensor(motion).float()) # (L,23*3)
        else:
            motion_tensor = torch.tensor(motion)
        
        # numpy
        # if m_length < self.max_motion_length: # 固定输出动作长度为max_length ！！
        #     motion = np.concatenate([motion, np.zeros((self.max_motion_length - m_length, motion.shape[1])) ], axis=0)
        # torch
        if m_length < self.max_motion_length: # 固定输出动作长度为max_length    补0
            motion_tensor = torch.cat([motion_tensor, 
                                       torch.zeros((self.max_motion_length - m_length, motion_tensor.shape[1]), dtype=float)], axis=0)

        return motion_tensor

    def hml2smpl(self, motion):
        '''
        input：(l,263)x z y r 21*3 21*6 22*3 4
        return： (l,23,3)
        '''
        L, dim = motion.shape
        n_joints = 22 if dim == 263 else 21

        rotations_6d = motion[:,67:193].reshape((-1, n_joints-1, 6))
        rotations_mat = rotation_conversions.rotation_6d_to_matrix(rotations_6d)
        rotations_axis_angle = rotation_conversions.matrix_to_axis_angle(rotations_mat)

        root_quat, root_xyz = recover_root_rot_pos(motion)
        root_axis_angle = rotation_conversions.quaternion_to_axis_angle(root_quat)

        smpl = torch.zeros((L, (n_joints+1)*3), dtype=float) # 22*3的旋转，1*3的根xyz
        smpl[:,:3] = root_axis_angle
        smpl[:,3:-3] = rotations_axis_angle.flatten(1,2)
        smpl[:,-3:] = root_xyz
        return smpl





