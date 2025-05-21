import options.option_transformer as option_trans
args = option_trans.get_args_parser()
from dataset import dataset_control
from models.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt





if __name__ == '__main__':
    fixseed(args.seed)
    args.max_samples = 10000
    args.guidance_param = 2.5
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    args.eval_mode = 'no_mm'
    # if args.resume_root is None:  # 没有1阶段，就是2阶段纯文本
    #     args.eval_mode = 'with_mm'

    assert args.resume_trans is not None, 'Must specify resume_trans'
    assert args.S1_diffusion_step >0 and args.S2_diffusion_step >0
    
    # 计算multimodality：batchsize=32，test split有145次迭代，里面挑选3次迭代，每次迭代里同一个文本都跑30次生成，即得到了(96，30, dim)的动作特征。从(96,30,dim)里选取两次(96,10,dim)，计算距离，即为multimodality
    args.normalize_traj = True # 归一化轨迹再输入
    if args.eval_mode == 'no_mm':
        num_samples_limit = args.max_samples
        run_mm = False
        mm_num_samples = 0 #  100
        mm_num_repeats = 0 # 一个文本生成几次动作, 30次
        mm_num_times = 0 # 10   算multimodality
        diversity_times = 300
        replication_times = args.replication_times # 重复测试次数
    elif args.eval_mode == 'with_mm':
        num_samples_limit = args.max_samples
        run_mm = True
        mm_num_samples = 100 #  100
        mm_num_repeats = 30 # 一个文本生成几次动作, 30次
        mm_num_times = 10 # 10   算multimodality
        diversity_times = 300
        replication_times = args.replication_times # 重复测试次数
    else:
        raise ValueError()


    if args.S1_diffusion_step < 1000 or args.S2_diffusion_step < 1000:
        assert args.use_ddim == 1, args.use_ddim

    if args.only_t2m_s2:
        log_file = f"{os.path.dirname(args.resume_trans)}/t2m"
    else:
        log_file = f"{os.path.dirname(args.resume_trans)}/joint_{str(args.control_joint).replace(' ','')}_density_{args.density}"

    log_file += f'_repeat{args.replication_times}'
    if args.S1_diffusion_step != 1000 or args.S2_diffusion_step != 1000:
        log_file += f'_{args.S1_diffusion_step}_{args.S2_diffusion_step}'
    
    if args.use_lbfgs and (not args.gtric_fortest)  and (not args.only_t2m_s2):
        log_file += f"_bfgslr_{str(args.bfgs_lr).replace('.','')}"
    if args.guidance_param != 2.5:
        log_file += f'_scale{args.guidance_param}'
    if args.bfgs_type !=0 and (not args.only_t2m_s2) and args.use_lbfgs:
        log_file += f'_bfgstype{args.bfgs_type}'
    log_file += f'_num{args.max_samples}'
    log_file += '.log'
    
    if sys.gettrace():
        log_file = f'output/debug/1.log'
    logger = get_logger('', file_path=log_file)
    logger.info(f'*************************************************************')
    # logger.info(f'gtric 文本长度77')
    logger.info(f'log_file = {log_file}')
    logger.info(f'args.dataset_name = {args.dataset_name}')
    logger.info(f'args.roottype = {args.roottype}')
    logger.info(f'args.resume_root = {args.resume_root}')
    logger.info(f'args.modeltype = {args.modeltype}')
    logger.info(f'args.resume_trans = {args.resume_trans}')
    logger.info(f'control joint = {args.control_joint}, density = {args.density}')
    logger.info(f'args.cond_mode = {args.cond_mode}')
    logger.info(f'args.guidance_param = {args.guidance_param}')
    logger.info(f'args.use_lbfgs = {args.use_lbfgs}')
    logger.info(f'args.replication_times = {args.replication_times}')
    logger.info(f'args.S1_diffusion_step = {args.S1_diffusion_step}')
    logger.info(f'args.S2_diffusion_step = {args.S2_diffusion_step}')
    logger.info(f'args.gtric_fortest = {args.gtric_fortest}')
    logger.info(f'args.only_t2m_s2 = {args.only_t2m_s2}')
    logger.info(f'args.bfgs_lr = {args.bfgs_lr}')
    logger.info(f'args.bfgs_type = {args.bfgs_type}')
    
    

    # CLIP
    clip_model = get_clip_model()

    # 根节点网络
    diffusion_root = None
    if args.roottype is not None:
        if args.roottype in ['omni67mdm_spatial', 'omni263mdm_spatial']: 
            from models.omnimdm_spatial import CMDM
            from utils.model_util import get_omni67_args
            net_root = CMDM(**get_omni67_args(args, args.roottype))
        else:
            raise NotImplementedError
        
        try:
            load_ckpt(net_root, args.resume_root, key='trans')
        except:
            load_ckpt(net_root, args.resume_root, key=None, strict=False)
            
        diffusion_root = create_gaussian_diffusion_simple(args, net_root, args.roottype, clip_model)
        net_root.cuda()
        net_root.eval()

    # 2阶段网络
    if args.modeltype in ['diffmdm', 'mdm']:
        from utils.model_util import get_mdm_args
        from models.mdm import MDM
        net = MDM(**get_mdm_args(args))
    else:
        raise NotImplementedError

    load_ckpt(net, args.resume_trans, key=None, strict=True)

    if args.guidance_param != 1:
        net = ClassifierFreeSampleModel(net)
    else:
        logger.info('NO CFG !!!!!!!!!!!!!!')
            
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype, clip_model)
    net.cuda()
    net.eval()
    

    #评估生成数据集部分  shuffle = False
    gt_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='gt', split='test', shuffle=True, num_workers=0, drop_last=True)
    gen_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=True, num_workers=0, drop_last=True)
    eval_motion_loaders = {
        ## HumanML3D Dataset##
        'vald': lambda: get_control_dataset(
            args, gen_loader, clip_model, diffusion_root, diffusion, mm_num_samples, mm_num_repeats, num_samples_limit
        )
    }
    eval_wrapper = EvaluatorMDMWrapper(args.dataset_name, torch.device('cuda'))
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)

    # 假设 feat_x0: [N, 512], feat_predict: [N, 512]
    all_feats = np.concatenate([feat_x0, feat_predict], axis=0)
    labels = ['GT'] * len(feat_x0) + ['Gen'] * len(feat_predict)

    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    embedding = tsne.fit_transform(all_feats)

    plt.scatter(embedding[:,0], embedding[:,1], c=['r' if l=='GT' else 'b' for l in labels])
    plt.title("t-SNE of motion encoder features")
    plt.legend(["GT", "Generated"])
    plt.show()
