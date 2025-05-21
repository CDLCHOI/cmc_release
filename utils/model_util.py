
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from diffusion.gaussian_diffusion_simple import GaussianDiffusionSimple
from utils.mask_utils import TextCLIP, calc_loss_xyz, vis_motion, calc_err_perframe, calc_loss_xyz_perbatch
import clip
import torch
import logging
from os.path import join as pjoin
from sys import stdout
import ipdb
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
    if modeltype in ['omni67mdm_spatial', 'critic', 'mdmcritic']:
        args.timestep_respacing = str(args.S1_diffusion_step)
    elif modeltype in ['mdm', 'diffmdm',]:
        args.timestep_respacing = str(args.S2_diffusion_step)
    else:
        print('modeltype = ', modeltype)
        raise NotImplementedError
        
    scale_beta = 1.  # no scaling
    steps = 1000
    betas = gd.get_named_beta_schedule('cosine', steps, scale_beta)


    return GaussianDiffusionSimple(args, model, modeltype, clip_model, betas)

def sample_omnicontrol(diffusion_root, args, model_kwargs, vis=False):
    
    # stage1
    pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=model_kwargs,batch_size=args.batch_size)


    if args.normalize_traj:
        traj = (model_kwargs['traj'] * diffusion_root.raw_std + diffusion_root.raw_mean) * model_kwargs['traj_mask']

    if vis:
        joint1, joint2 = vis_motion(pred_ric[0][..., :67], model_kwargs['gt_motion'][0][..., :67], dataset=args.dataset_name, save_path=f'./output/testsample/stage1.html', joints2_from_rot=False)
    loss_xyz, err = calc_loss_xyz(pred_ric, traj, model_kwargs['traj_mask']) # 仅约束控制的关节误差

    return pred_ric, loss_xyz, err

def sample_ADControl(diffusion_root, diffusion, args, model_kwargs, vis=False, only_stage1=False, filename=None):
    """sample for both stage1 and stage2. Only suppert HumanML3D dataset
    """

    # stage1
    if args.gtric_fortest:
        pred_ric = model_kwargs['gt_ric']
    else:
        pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=model_kwargs,batch_size=args.batch_size)
    # if vis:
    #     vis_motion(pred_ric[0], model_kwargs['gt_ric'][0], dataset=args.dataset_name, save_path=f'./output/testsample/stage1.html')

    if args.normalize_traj:
        traj = (model_kwargs['traj'] * diffusion_root.raw_std + diffusion_root.raw_mean) * model_kwargs['traj_mask']
    # if vis:
    #     joint1, joint2 = vis_motion(pred_ric[0][..., :67], model_kwargs['gt_motion'][0][..., :67], dataset=args.dataset_name, save_path=f'./output/testsample/stage1.html', joints2_from_rot=False)
        
    calc_loss_xyz_perbatch(pred_ric, traj)
    loss_xyz, err = calc_loss_xyz(pred_ric, traj, model_kwargs['traj_mask']) # 仅约束控制的关节误差
    _, err_perframe = calc_err_perframe(pred_ric, traj, model_kwargs['traj_mask'])
    # plt.clf(); plt.ylim(-0.2,0.5); plt.plot(err_perframe[0]); plt.title('err per frame'); plt.savefig('err_per_frame.png')
    if only_stage1:
        if vis:
            joint1, joint2 = vis_motion(pred_ric[0][..., :67], model_kwargs['gt_motion'][0][..., :67], dataset=args.dataset_name, save_path=f'./output/testsample/{filename}.html', joints2_from_rot=False)
        return pred_ric, loss_xyz, err 

    

    # stage2
    sample = []
    for j in range(args.stage2_repeat_times):
        partial_emb = torch.zeros_like(model_kwargs['gt_motion'], device=model_kwargs['gt_motion'].device)
        partial_emb[..., :67] = pred_ric  
        pred_motion = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=model_kwargs, batch_size=args.batch_size) 
        savename = f'./output/testsample/{filename}.html'
        if vis:
            joint1, joint2 = vis_motion(pred_motion[0], model_kwargs['gt_motion'][0], dataset=args.dataset_name, save_path=savename, joints2_from_rot=False)
        sample.append(pred_motion)
    
    return sample, loss_xyz, err

def sample_omni263mdm_fuse(diffusion, args, model_kwargs, vis=False, only_stage1=False):
    """sample for both stage1 and stage2. Only suppert HumanML3D dataset
    """
    # stage1
    pred_ric = diffusion.p_sample_loop(partial_emb=None, model_kwargs=model_kwargs,batch_size=args.batch_size, indices_bound=None) # (b,196,263)

    # if vis:
    #     vis_motion(pred_ric[0], model_kwargs['gt_ric'][0], dataset=args.dataset_name, save_path=f'./output/testsample/stage1.html')

    if args.normalize_traj:
        traj = (model_kwargs['traj'] * diffusion.raw_std + diffusion.raw_mean) * model_kwargs['traj_mask']
    if vis:
        joint1, joint2 = vis_motion(pred_ric[0][..., :67], model_kwargs['gt_motion'][0][..., :67], dataset=args.dataset_name, save_path=f'./output/testsample/stage1.html', joints2_from_rot=False)
    loss_xyz, err = calc_loss_xyz(pred_ric, traj, model_kwargs['traj_mask']) # 仅约束控制的关节误差
    if only_stage1:
        return pred_ric, loss_xyz, err



    # stage2
    partial_emb = torch.zeros_like(model_kwargs['gt_motion'], device=model_kwargs['gt_motion'].device)
    partial_emb = pred_ric  
    pred_motion = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=model_kwargs, batch_size=args.batch_size) 
    if vis:
        joint1, joint2 = vis_motion(pred_motion[0], model_kwargs['gt_motion'][0], dataset=args.dataset_name, save_path=f'./output/testsample/stage2.html', joints2_from_rot=False)
    
    return pred_motion, loss_xyz, err

def get_mdmcritic_args(args):
    assert args.dataset_name == 't2m', args.dataset_name
    if args.datatype == 'hml':
        # (b,196,1,263)
        njoints = 263
        nfeats = 1
    elif args.datatype == 'smpl':
        # (b,196,23,3)
        njoints = 22 + 1
        nfeats = 3
    

    return {'args':args, 'njoints': njoints, 'nfeats': nfeats, 
            'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu"}


def get_mdm_args(args, modeltype=None):

    # default args
    clip_version = 'ViT-B/32'
    cond_mode = 'text'

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset_name == 't2m':
        data_rep = 'hml_vec'
        if modeltype == 'mdm67_spatial':
            njoints = 67
        else:
            njoints = 263
        
        nfeats = 1
    elif args.dataset_name == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    return {'modeltype': modeltype, 'njoints': njoints, 'nfeats': nfeats, 'num_actions': 1,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': 0.1, 'arch': 'trans_enc', 'clip_version': 'ViT-B/32',
            'dataset': args.dataset_name}

def get_omni67_args(args, modeltype=None):


    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset_name == 't2m':
        data_rep = 'hml_vec'
        if modeltype in ['omni67', 'omni67res', 'omni67mdm_spatial', 'mdm67_spatial']:
            njoints = 67
        elif modeltype == 'omni193':
            njoints = 193
        elif modeltype == 'omni259':
            njoints = 259
        elif modeltype in ['omnicontrol', 'omni263mdm_fuse', 'omni263mdm_spatial']:
            njoints = 263
        else:
            raise NotImplementedError
        nfeats = 1
    elif args.dataset_name == 'kit':
        data_rep = 'hml_vec'
        if modeltype in ['omni67', 'omni67res', 'omni67mdm_spatial', 'mdm67_spatial']:
            njoints = 64
        elif modeltype in ['omnicontrol', 'omni263mdm_fuse', 'omni263mdm_spatial']:
            njoints = 251
        else:
            raise NotImplementedError
        nfeats = 1
    # ipdb.set_trace()
    print('njoints = ', njoints)

    return {'modeltype': modeltype, 'njoints': njoints, 'nfeats': nfeats, 'num_actions': 1,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': args.cond_mode,
            'cond_mask_prob': 0.1, 'arch': 'trans_enc', 'clip_version': 'ViT-B/32',
            'dataset': args.dataset_name}

def get_semanticboost_args(args, modeltype=None):

    clip_version = 'ViT-B/32'
    args.arch = "llama_decoder_rope"
    # cond_mode = 'no_cond'
    cond_mode = "text"
     
    activation = "swiglu"

    if args.dataset_name == 't2m':
        if modeltype in ['semboost', 'semboost_s1', 'semboost22']:
            njoints = 263
        elif modeltype == 'semboost_4':
            njoints = 4
        elif modeltype == 'semboost_67':
            njoints = 67 
        else:
            raise NotImplementedError
        nfeats = 1
        dataset = "humanml"
    elif args.dataset_name == 'kit':
        njoints = 251
        nfeats = 1
        dataset = "kit"

    return {'modeltype':modeltype, 'njoints': njoints, 'nfeats': nfeats, 'latent_dim': 512, 'ff_size': 1024, 
            'num_layers': 8, 'num_heads': 4,
            'dropout': 0.1, 'activation': activation, 'cond_mode': cond_mode, 'cond_mask_prob': 0.1, 'arch': args.arch,
            'clip_version': clip_version, 'dataset': dataset, "local":False, "encode_full":2, "txt_tokens":2,
            "num_frames":196, "conv_bias":True, "conv_activate":"relu", 
            "conv_norm":"layernorm"}

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

######################################################################
#########################  for original omnicontrol  #################
######################################################################

def get_omnicontrol_args(args):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset_name == 't2m':
        data_rep = 'hml_vec'
        njoints = 263 # + 66
        nfeats = 1
    elif args.dataset == 'kit':
        data_rep = 'hml_vec'
        njoints = 251
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': 1,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': 'both_text_spatial',
            'cond_mask_prob': 0.1, 'arch': 'trans_enc', 'clip_version': 'ViT-B/32',
            'dataset': args.dataset_name}


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


def create_model_and_diffusion(args, data, modeltype='CMDM'):
    assert modeltype in ['CMDM', 'CMDM67']
    if modeltype == 'CMDM':
        from models.omnicontrol import CMDM
        model = CMDM(**get_omnicontrol_args(args, data))
    elif modeltype == 'CMDM67':
        from models.omni67 import CMDM
        model = CMDM(**get_omnicontrol_args(args, data))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion
