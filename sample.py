import options.option_transformer as option_trans
import os 
args = option_trans.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
from utils.fixseed import fixseed
# fixseed(123456) # 站着挥右手
# fixseed(111) # 直走路
fixseed(3722241)
from dataset import dataset_control
from utils.mask_utils import generate_src_mask, load_ckpt
from utils.model_util import create_gaussian_diffusion_simple, get_clip_model, sample_ADControl, sample_omni263mdm_fuse, sample_omnicontrol
from utils.text_control_example import collate_all
import clip
import copy
from models.cfg_sampler import ClassifierFreeSampleModel, CFG2, CFG3

if __name__ == '__main__':
    args.dataset_name = 't2m'
    args.stage2_repeat_times = 1
    args.control_joint = [0]
    args.density = 100
    args.use_lbfgs = 1
    args.bfgs_type = 2
    args.normalize_traj=True
    args.batch_size = 1

    args.S1_diffusion_step = 1000
    args.S2_diffusion_step = 1000
    # args.use_ddim = 1; args.S1_diffusion_step = 100; args.S2_diffusion_step = 100
    args.return_type = 'sample'
    args.stage2_no_root_y = 0
    # args.gtric_fortest = True; print('!!!!!!!!!! use gtric for test')

    ### stage 1
    # args.resume_root = 'output/0518_omni67_multi_partxyz/net_last.pth'; args.roottype = 'omni67'; args.normalize_traj=True; args.cond_mode = 'no_cond'
    # args.resume_root = 'output/0627_omni67_textcond/net_best.pth'; args.roottype = 'omni67'; args.normalize_traj=True; args.cond_mode = 'both_text_spatial'; key='trans'
    # args.resume_root = 'output/0820_omni67res/net_last.pth'; args.roottype = 'omni67res'; args.normalize_traj=True; args.cond_mode = 'both_text_spatial'; key=None
    # args.resume_root = 'output/0822_omni67res_trainposterior/net_last.pth'; args.roottype = 'xxxx'; args.normalize_traj=True; args.cond_mode = 'both_text_spatial'; key=None
    # args.resume_root = 'output/0823_omni67mdm_spatial/net_last.pth'; args.roottype = 'omni67mdm_spatial'; args.normalize_traj=True; args.cond_mode = 'both_text_spatial'; key=None
    # args.resume_root = 'output/0826_omni67mdm_spatial_bfgs10/net_last.pth'; args.roottype = 'omni67mdm_spatial'; args.normalize_traj=True; args.cond_mode = 'both_text_spatial'; key=None  # √
    # args.resume_root = '/home/deli/project/ADControl/output/omnicontrol_ckpt/model_humanml3d.pt'; args.roottype = 'omnicontrol'; args.normalize_traj=True; args.cond_mode = 'both_text_spatial'; key=None; args.use_lbfgs = 0
    # args.resume_root = 'output/0902_omni67mdm_spatial_bfgs_randomnohint/net_last.pth'; args.roottype = 'omni67mdm_spatial'; args.normalize_traj=True; args.cond_mode = 'both_text_spatial'; key=None; args.only_t2m_s1 = True
    
    # args.resume_root = 'output/0909_omni67mdm_spatial_only_t2m/net_last.pth'; args.roottype = 'omni67mdm_spatial'; args.normalize_traj=True; args.cond_mode = 'both_text_spatial'; key=None; args.only_t2m_s1 = True; args.control_joint = [0]
    # args.resume_root = 'output/0909_omni67mdm_spatial_only_t2m/net_last.pth'; args.roottype = 'mdm67_spatial'; args.normalize_traj=True; args.cond_mode = 'both_text_spatial'; key=None; args.only_t2m_s1 = True; args.control_joint = [0]
    args.resume_root = 'output/1009_omni67mdm_spatial_bfgs/net_last.pth'; args.roottype = 'omni67mdm_spatial'; args.normalize_traj=True; args.cond_mode = 'both_text_spatial'; key=None
    # args.resume_root = 'output/1025_omnimdm_spatial_bfgs07_kit/net_last.pth'; args.roottype = 'omni67mdm_spatial'; args.normalize_traj=True; args.cond_mode = 'both_text_spatial'; key=None

    ### stage 2
    key2 = 'trans'
    # args.resume_trans = 'output/0520_semboost/net_last.pth'; args.modeltype = 'semboost'
    args.resume_trans = 'output/0816_diffmdm/net_last.pth'; args.modeltype = 'diffmdm'; args.return_type = 'sample'
    # args.resume_trans = 'output/0816_diffmdm/net_last.pth'; args.modeltype = 'omnimdm'; args.return_type = 'sample'
    # args.resume_trans = 'output/0913_diffmdm/net_last.pth'; args.modeltype = 'diffmdm'
    # args.resume_trans = 'output/0912_diffmdm_mask/net_last.pth'; args.modeltype = 'diffmdm_mask'; key2=None
    # args.resume_trans = 'output/0913_diffmdm_mask/net_last.pth'; args.modeltype = 'diffmdm_mask'
    # args.resume_trans = 'output/0912_diffmdm_mask_res/net_last.pth'; args.modeltype = 'diffmdm_mask_res'
    # args.resume_trans = 'output/0913_diffmdm_mask_res/net_last.pth'; args.modeltype = 'diffmdm_mask_res'
    # args.resume_trans = 'output/0918_omnimdm_mask/net_last.pth'; args.modeltype = 'omnimdm_mask'
    # args.resume_trans = 'output/0918_omnimdm_mask_only_inpaint/net_last.pth'; args.modeltype = 'omnimdm_mask'
    # args.resume_trans = 'output/1025_diffmdm_kit/net_last.pth'; args.modeltype = 'diffmdm'



    clip_model = get_clip_model()

    # 根节点网络
    if args.roottype == 'omni67mdm_spatial': # 用omnicontrol的代码去掉controlnet部分，仅保留spatial_encoder
        from models.omnimdm_spatial import CMDM
        from utils.model_util import get_omni67_args
        net_root = CMDM(**get_omni67_args(args, args.roottype))
    else:
        raise NotImplementedError
    # load_ckpt(net_root, args.resume_root, key='trans', strict=True)
    load_ckpt(net_root, args.resume_root, key=key, strict=False)
    
    net_root.eval()
    net_root.cuda()
    diffusion_root = create_gaussian_diffusion_simple(copy.deepcopy(args), net_root, args.roottype, clip_model)


    # 2阶段网络
    if args.modeltype == 'diffmdm':
        from models.mdm import MDM
        from utils.model_util import get_mdm_args
        net = MDM(**get_mdm_args(args))
    elif args.modeltype == 'omnimdm': # omnicontrol的transformer搭的mdm
        from models.omnimdm import CMDM
        from utils.model_util import get_mdm_args
        net = CMDM(**get_mdm_args(args, args.modeltype))
    else:
        raise NotImplementedError
    
    load_ckpt(net, args.resume_trans, key=key2)
    # try:
    #     load_ckpt(net, args.resume_trans, key='trans')
    # except:
    #     load_ckpt(net, args.resume_trans, key=None)

    if args.guidance_param != 1:
        net = ClassifierFreeSampleModel(net)
    else:
        print(' NO CFG !!!!!!!!!!!!!!!!!!!!!')
    
                
    net.eval()
    net.cuda()
    diffusion = create_gaussian_diffusion_simple(copy.deepcopy(args), net, args.modeltype, clip_model)

    # diffusion2 = create_gaussian_diffusion_simple(args, net, 'semboost', clip_model)

    # create dataloader
    # train_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', shuffle=False,)
    # train_loader_iter = dataset_control.cycle(train_loader)
    # val_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='val', shuffle=True, num_workers=0)
    # val_loader_iter = dataset_control.cycle(val_loader)
    test_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=False, num_workers=0, drop_last=True)

    os.system('rm *.npy')

    
    for i, batch in enumerate(test_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
        b, max_length, num_features = gt_motion.shape
        gt_motion = gt_motion.cuda()
        real_length = real_length.cuda()
        traj = traj.cuda()
        traj_mask = traj_mask.cuda()
        traj_mask_263 = traj_mask_263.cuda()
        real_mask = generate_src_mask(max_length, real_length) # (b,196)
        gt_ric = gt_motion[..., :67]

        if args.stage2_no_root_y:
            traj_mask_263[..., 3] = False

        print('filename = ', filename)
        text = clip.tokenize(clip_text, truncate=True).cuda() 
        text_emb, word_emb = clip_model(text)
        print('clip_text = ', clip_text[0])

        condition = {}
        condition['traj'] = traj.clone()
        condition['traj_mask'] = traj_mask
        condition['gt_ric'] = gt_ric
        condition['traj_mask_263'] = traj_mask_263
        condition['gt_motion'] = gt_motion
        condition['real_mask'] = real_mask
        condition['clip_text'] = clip_text
        condition['text_emb'] = text_emb
        condition['word_emb'] = word_emb
        condition['real_length'] = real_length

        

        if args.roottype == 'omnicontrol':
            sample, loss_xyz, err1 = sample_omnicontrol(diffusion_root, args, condition, vis=True)    
        else:   
            sample, loss_xyz, err1 = sample_ADControl(diffusion_root, diffusion, args, condition, vis=True, only_stage1=True, filename=filename[0])
            # print(f'loss_xyz = {loss_xyz.item():.4f}')
        print(f'err1 = {err1:.4f}')

        break
        


        

    

    
    
    
    