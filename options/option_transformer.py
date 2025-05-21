import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--eval_during_train', action='store_true', default=False)
    parser.add_argument('--max_motion_length', type=int, default=196)
    parser.add_argument('--only_emb_loss', action='store_true', default=False)
    parser.add_argument('--emb_loss', type=float, default=0)
    parser.add_argument('--use_cache', action='store_true', default=False)
    parser.add_argument('--datatype', type=str, choices=['hml', 'smpl'], default='smpl')
    parser.add_argument('--num_noisy_timesteps', type=int, default=10)
    parser.add_argument('--sim', action='store_true', default=False)
    parser.add_argument('--embloss', type=int, default=1)
    parser.add_argument('--timestep_respacing', type=str, default='100')
    parser.add_argument('--use_ddim', type=int, choices=[1, 0], default=0)
    parser.add_argument('--only_t2m_s2', type=int, default=0)
    parser.add_argument('--bfgs_type', type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument('--guidance_param', type=float, default=2.5)
    parser.add_argument('--bfgs_lr', type=float, default=0.5) 
    parser.add_argument('--gtric_fortest', action='store_true', default=False)
    parser.add_argument('--S1_diffusion_step', type=int, default=1000)
    parser.add_argument('--S2_diffusion_step', type=int, default=1000)
    parser.add_argument('--root_zero_grad', type=int, default=0)
    parser.add_argument('--replication_times', type=int, default=1)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--train_posterior', action='store_true', default=False)
    parser.add_argument('--use_lbfgs', type=int, choices=[1, 0], default=0)
    parser.add_argument('--stage2_no_root_y', type=int, choices=[0, 1], default=0)
    parser.add_argument('--return_type', type=str, choices=['x0', 'sample', 'priorMDM'], default='sample')
    parser.add_argument("--max_samples", default=1000, type=int)
    parser.add_argument("--cond_mode", default='both_text_spatial', type=str)
    parser.add_argument('--loss_type', type=str, choices=['l1', 'l2'], default='l2')
    # parser.add_argument("--control_joint", default=-1, type=int, help='-1 means randomly choose a joint')
    parser.add_argument('--control_joint', default=[-1], nargs="+", type=int)
    parser.add_argument('--density',  type=int, default=0)
    parser.add_argument('--dense_control', action='store_true', default=False)
    parser.add_argument('--attn_mask', action='store_true', default=False)

    parser.add_argument('--root_dist_loss', action='store_true', default=False, help='instead of element-wise loss, use global loss for root')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'debug'], default='train')
    parser.add_argument('--multi_joint_control', action='store_true', default=False)
    parser.add_argument('--temporal_complete', type=float, default=0.0, help='whether add temporal completion')
    parser.add_argument('--normalize_traj', action='store_true', default=True)
    parser.add_argument('--note', type=str, default='this is note')
    parser.add_argument('--num_layers_E', type=int, default=3)
    parser.add_argument('--num_layers_D', type=int, default=1)
    parser.add_argument('--roottype', type=str, default=None) 
    parser.add_argument('--modeltype', type=str, default=None) 
    parser.add_argument('--gpu', nargs='+', default=['0'])
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")

    ## dataloader
    parser.add_argument('--dataset_name', type=str, default='t2m', choices=['t2m', 'kit'], help='dataset directory')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    
    ## optimization
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[150000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    # training settings
    parser.add_argument("--resume_root", type=str, default=None, help='resume gpt pth')
    parser.add_argument("--resume_trans", type=str, default=None, help='resume gpt pth')
    parser.add_argument('--out_dir', type=str, default='output', help='output directory')
    parser.add_argument('--exp_name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out_dir')
    parser.add_argument('--print_iter', default=20, type=int, help='print frequency')
    parser.add_argument('--eval_iter', default=10000, type=int, help='evaluation frequency')
    parser.add_argument('--save_iter', default=10000, type=int, help='save frequency')
    parser.add_argument('--total_iter', default=300000, type=int, help='number of total iterations to run')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")

    ## vqvae arch
    # parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    # parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    # parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    # parser.add_argument("--width", type=int, default=512, help="width of the network")
    # parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    # parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    # parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    # parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')

    ## gpt arch
    # parser.add_argument("--block-size", type=int, default=51, help="seq len")
    # parser.add_argument("--embed-dim-gpt", type=int, default=1024, help="embedding dimension")
    # parser.add_argument("--num-layers", type=int, default=9, help="nb of transformer layers")
    # parser.add_argument("--num-local-layer", type=int, default=1, help="nb of transformer local layers")
    # parser.add_argument("--n-head-gpt", type=int, default=16, help="nb of heads")
    # parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    # parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")
    
    ## quantizer
    # parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    # parser.add_argument('--quantbeta', type=float, default=1.0, help='dataset directory')

    ## resume
    # parser.add_argument("--resume-pth", type=str, default=None, help='resume vq pth')
    
    
    ## output directory 
    
    # parser.add_argument('--vq-name', type=str, default='exp_debug', help='name of the generated dataset .npy, will create a file inside out_dir')
    ## other
    
    # parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    # parser.add_argument('--pkeep', type=float, default=.5, help='keep rate for gpt training')
    
    ## generator
    parser.add_argument('--text', type=str, help='text')
    parser.add_argument('--length', type=int, help='length')

    return parser.parse_args()