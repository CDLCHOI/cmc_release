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
from utils.model_util import create_gaussian_diffusion_simple, get_clip_model, sample_cmc
import clip
import copy
from models.cfg_sampler import ClassifierFreeSampleModel

if __name__ == '__main__':
    args.dataset_name = 't2m'
    args.control_joint = [0]
    args.density = 100
    args.batch_size = 1
    # args.S1_diffusion_step = 1000
    # args.S2_diffusion_step = 1000
    args.use_ddim = 1; args.S1_diffusion_step = 100; args.S2_diffusion_step = 100


    ### stage 1
    args.resume_root = 'output/cmc_humanml3d/cmc_humanml3d_s1.pth'; args.roottype = 's1'
    ### stage 2
    args.resume_trans = 'output/cmc_humanml3d/cmc_humanml3d_s2.pth'; args.modeltype = 's2'



    clip_model = get_clip_model()

    # S1
    from models.omnimdm_spatial import CMDM
    from utils.model_util import get_s1_args
    net_root = CMDM(**get_s1_args(args, args.roottype))
    load_ckpt(net_root, args.resume_root)
    
    net_root.eval()
    net_root.cuda()
    diffusion_root = create_gaussian_diffusion_simple(copy.deepcopy(args), net_root, 's1', clip_model)


    # S2
    from models.mdm import MDM
    from utils.model_util import get_s2_args
    net = MDM(**get_s2_args(args))
    load_ckpt(net, args.resume_trans)
    net = ClassifierFreeSampleModel(net)  
    net.eval()
    net.cuda()
    diffusion = create_gaussian_diffusion_simple(copy.deepcopy(args), net, 's2', clip_model)


    # create dataloader
    test_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=False, num_workers=0, drop_last=True)

    
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

        

        sample, loss_xyz, err = sample_cmc(diffusion_root, diffusion, args, condition, vis=True, filename=filename[0])
        print(f'err = {err:.4f}')
        break
        


        

    

    
    
    
    