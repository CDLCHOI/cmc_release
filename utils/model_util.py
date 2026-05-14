
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.gaussian_diffusion_simple import GaussianDiffusionSimple
from utils.mask_utils import TextCLIP, calc_loss_xyz, vis_motion, calc_err_perframe, calc_loss_xyz_perbatch
import clip
import torch
import logging
from os.path import join as pjoin
from sys import stdout
import matplotlib.pyplot as plt

def get_clip_model():
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model = TextCLIP(clip_model)
    return clip_model

def create_gaussian_diffusion_simple(args, model, modeltype, clip_model=get_clip_model()):
    if modeltype == 's1':
        args.timestep_respacing = str(args.S1_diffusion_step)
    elif modeltype == 's2':
        args.timestep_respacing = str(args.S2_diffusion_step)
    else:
        raise NotImplementedError(f'{modeltype}')
        
    scale_beta = 1.  # no scaling
    steps = 1000
    betas = gd.get_named_beta_schedule('cosine', steps, scale_beta)
    return GaussianDiffusionSimple(args, model, modeltype, clip_model, betas)


def sample_cmc(diffusion_root, diffusion, args, model_kwargs, vis=False, only_stage1=False, filename=None):
    """sample for both stage1 and stage2. Only support HumanML3D dataset
    """
    assert args.dataset_name == 't2m', args.dataset_name
    savename = f'./output/testsample/{filename}.html'

    # stage1
    pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=model_kwargs,batch_size=args.batch_size)

    # calculate error
    traj = (model_kwargs['traj'] * diffusion_root.raw_std + diffusion_root.raw_mean) * model_kwargs['traj_mask']
    calc_loss_xyz_perbatch(pred_ric, traj)
    loss_xyz, err = calc_loss_xyz(pred_ric, traj, model_kwargs['traj_mask'])
    _, err_perframe = calc_err_perframe(pred_ric, traj, model_kwargs['traj_mask'])
    if only_stage1:
        if vis:
            joint1, joint2 = vis_motion(pred_ric[0][..., :67], model_kwargs['gt_motion'][0][..., :67], dataset=args.dataset_name, save_path=savename)
            print(f'save {savename}')
        return pred_ric, loss_xyz, err 

    

    # stage2
    partial_emb = torch.zeros_like(model_kwargs['gt_motion'], device=model_kwargs['gt_motion'].device)
    partial_emb[..., :67] = pred_ric  
    pred_motion = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=model_kwargs, batch_size=args.batch_size) 
    
    if vis:
        joint1, joint2 = vis_motion(pred_motion[0], model_kwargs['gt_motion'][0], dataset=args.dataset_name, save_path=savename)
        print(f'save {savename}')
    
    return pred_motion, loss_xyz, err



def get_s2_args(args, modeltype=None):

    # SMPL defaults
    data_rep = 'hml_vec'

    if args.dataset_name == 't2m':
        njoints = 263
        nfeats = 1
    elif args.dataset_name == 'kit':
        njoints = 251
        nfeats = 1

    return {'modeltype': modeltype, 'njoints': njoints, 'nfeats': nfeats, 'num_actions': 1,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': args.cond_mode,
            'cond_mask_prob': 0.1, 'arch': 'trans_enc', 'clip_version': 'ViT-B/32',
            'dataset': args.dataset_name}

def get_s1_args(args, modeltype=None):
    data_rep = 'hml_vec'

    if args.dataset_name == 't2m':
        njoints = 67
        nfeats = 1
    elif args.dataset_name == 'kit':
        njoints = 64
        nfeats = 1

    return {'modeltype': modeltype, 'njoints': njoints, 'nfeats': nfeats, 'num_actions': 1,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': args.cond_mode,
            'cond_mask_prob': 0.1, 'arch': 'trans_enc', 'clip_version': 'ViT-B/32',
            'dataset': args.dataset_name}


def initial_optim(lr, weight_decay, net, optimizer) : 
    
    if optimizer == 'adamw' : 
        optimizer_adam_family = torch.optim.AdamW
    elif optimizer == 'adam' : 
        optimizer_adam_family = torch.optim.Adam
    
    optimizer = optimizer_adam_family(net.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=weight_decay)
        
    return optimizer

def get_logger(out_dir, file_path=None):
    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    if file_path == None:
        file_path = pjoin(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger



def create_gaussian_diffusion(args):
    args.noise_schedule = 'cosine'
    args.sigma_small = True
    args.lambda_vel = 0.0
    args.lambda_rcxyz = 0.0
    args.lambda_fc = 0.0

    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = 1000
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        dataset=args.dataset_name
    )
