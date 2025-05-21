import os 
import sys
import options.option_transformer as option_trans
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)  # 设定GPU
# os.environ['OMP_NUM_THREADS'] = '8'

import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import numpy as np
import ipdb

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import clip
import json
from utils.model_util import initial_optim, get_logger
from utils.mask_utils import load_ckpt
from dataset import dataset_control, dataset_critic
import warnings
warnings.filterwarnings('ignore')
import shutil

if __name__ == '__main__':
    # 训练前准备
    args.out_dir = pjoin(args.out_dir, args.exp_name) # output/trans_exp_name
    if args.overwrite and os.path.exists(args.out_dir):
        assert not os.path.exists(pjoin(args.out_dir, 'net_last.pth')), f'net_last.pth exist in {args.out_dir}'
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok = True)

    # logger
    logger = get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True)) # args所有输出到log
    logger.info(args.note)
    torch.manual_seed(args.seed)

    # mean and std
    humanml_mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy')).cuda()[None, None, ...] # dataset/HumanML3D/Mean.npy
    humanml_std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy')).cuda()[None, None, ...]
    
    # CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    # clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
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
    clip_model = TextCLIP(clip_model)

    # VA-VAE
    if args.modeltype == 'diffmdm':
        from models.mdm import MDM
        from utils.model_util import get_mdm_args
        net = MDM(**get_mdm_args(args, args.modeltype))
    elif args.modeltype in ['omni67mdm_spatial', 'omni263mdm_spatial']: # 用omnicontrol的代码去掉controlnet部分，仅保留spatial_encoder
        from models.omnimdm_spatial import CMDM
        from utils.model_util import get_omni67_args
        net = CMDM(**get_omni67_args(args, args.modeltype))
    elif args.modeltype in ['mdm']:
        from models.mdm import MDM
        from utils.model_util import get_mdm_args
        net = MDM(**get_mdm_args(args, args.modeltype))
    elif args.modeltype in ['critic']:
        from models.critic.critic import MotionCritic
        net = MotionCritic(depth=1, dim_feat=256, dim_rep=512, mlp_ratio=4, num_joints=22+1 if args.dataset_name == 't2m' else 21+1)
    elif args.modeltype in ['mdmcritic']:
        from models.mdm_critic import MDMCritic
        from utils.model_util import get_mdmcritic_args
        net = MDMCritic(**get_mdmcritic_args(args))
    else:   
        raise ValueError("modeltype not found")

    from utils.model_util import create_gaussian_diffusion_simple
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype, clip_model)


    # load_ckpt(net, args.resume_trans, key='trans', strict=False)
    load_ckpt(net, args.resume_trans, key=None, strict=True)
    # except:
    #     load_ckpt(net, args.resume_trans, key=None, strict=False)
            
    if sys.gettrace():
        net.eval(); logger.info(' net is eval !!!!!!!')
    else:
        net.train(); logger.info(' net is train ~~~~~')

    net = nn.DataParallel(net, device_ids=list(range(0,len(args.gpu))))
    net.cuda()

    if args.modeltype in ['critic', 'mdmcritic']:
        train_loader = dataset_critic.DataLoader(args, diffusion, split='train')
        train_loader_iter = dataset_critic.cycle(train_loader)
    else:
        train_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode=args.mode)
        train_loader_iter = dataset_control.cycle(train_loader)
        gt_loader = dataset_control.DataLoader(batch_size=32, args=args, mode='gt', split='test', shuffle=True, num_workers=0, drop_last=True)
        gen_loader = dataset_control.DataLoader(batch_size=32, args=args, mode='eval', split='test', shuffle=True, num_workers=0, drop_last=True)

    # 训练配置
    optimizer = initial_optim(args.lr, args.weight_decay, net, args.optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    
    if args.modeltype in ['omni67', 'omni67mdm_spatial']:
        if args.dataset_name == 't2m':
            d = 67
        elif args.dataset_name == 'kit':
            d = 64
        diffusion.trainer_func_s1(train_loader_iter, logger, optimizer, scheduler, dim=d)
    elif args.modeltype in ['diffmdm']:
        diffusion.trainer_func_s2(train_loader_iter, logger, optimizer, scheduler)
    elif args.modeltype in ['mdm']:
        print('log_file = ', os.path.join(args.out_dir, 'run.log'))
        diffusion.log_file = os.path.join(args.out_dir, 'run.log')
        diffusion.gt_loader = gt_loader
        diffusion.gen_loader = gen_loader
        diffusion.trainer_func_mdm(train_loader_iter, logger, optimizer, scheduler, writer)
    elif args.modeltype in ['critic', 'mdmcritic']:
        from models.critic import critic_trainer
        critic_trainer.trainer_func(net, train_loader_iter, logger, optimizer, scheduler, args)

    
    

