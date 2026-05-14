import os 
import sys
import options.option_transformer as option_trans
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
import torch
torch.backends.cudnn.enabled = False
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import clip
import json
from utils.model_util import initial_optim, get_logger
from utils.mask_utils import load_ckpt
from dataset import dataset_control
import warnings
warnings.filterwarnings('ignore')
import shutil

if __name__ == '__main__':
    args.out_dir = pjoin(args.out_dir, args.exp_name)
    if args.overwrite and os.path.exists(args.out_dir):
        assert not os.path.exists(pjoin(args.out_dir, 'net_last.pth')), f'net_last.pth exist in {args.out_dir}'
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok = True)

    # logger
    logger = get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info("python " + " ".join(sys.argv))
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

    
    if args.modeltype == 's1':
        from models.omnimdm_spatial import CMDM
        from utils.model_util import get_s1_args
        net = CMDM(**get_s1_args(args, args.modeltype))
    elif args.modeltype == 's2':
        from models.mdm import MDM
        from utils.model_util import get_s2_args
        net = MDM(**get_s2_args(args, args.modeltype))
    else:   
        raise ValueError("modeltype not found")

    from utils.model_util import create_gaussian_diffusion_simple
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype, clip_model)
    load_ckpt(net, args.resume_trans, key=None, strict=False)
            
    net.train(); logger.info(' net is train ~~~~~')

    net = nn.DataParallel(net, device_ids=list(range(0,len(args.gpu))))
    net.cuda()

    train_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode=args.mode)
    train_loader_iter = dataset_control.cycle(train_loader)
    gt_loader = dataset_control.DataLoader(batch_size=32, args=args, mode='gt', split='test', shuffle=True, num_workers=0, drop_last=True)
    gen_loader = dataset_control.DataLoader(batch_size=32, args=args, mode='eval', split='test', shuffle=True, num_workers=0, drop_last=True)

    optimizer = initial_optim(args.lr, args.weight_decay, net, args.optimizer)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

    
    if args.modeltype == 's1':
        diffusion.trainer_func_s1(train_loader_iter, logger, optimizer, scheduler)
    elif args.modeltype == 's2':
        diffusion.trainer_func_s2(train_loader_iter, logger, optimizer, scheduler)

    
    

