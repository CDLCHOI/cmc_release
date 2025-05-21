
import os
import options.option_transformer as option_trans
args = option_trans.get_args_parser()

from utils.mask_utils import load_ckpt
from models.critic.critic import MotionCritic
from utils.model_util import create_gaussian_diffusion_simple
from utils.mdm_plot_script import plot_3d_motion
from utils.motion_process import recover_from_smpl, recover_from_ric
from dataset import dataset_critic
import data_loaders.humanml.utils.paramUtil as paramUtil


if __name__ == '__main__':
    args.batch_size = 1
    args.datatype = 'hml'
    args.modeltype = 'mdmcritic'
    net = MotionCritic(depth=1, dim_feat=256, dim_rep=512, mlp_ratio=4, num_joints=22+1 if args.dataset_name == 't2m' else 21+1)
    net.eval()
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype, None)
    train_loader = dataset_critic.DataLoader(args, diffusion, split='train', shuffle=True)
    train_loader_iter = dataset_critic.cycle(train_loader)
    load_ckpt(net, 'output/0519_mdmcritic/net_last.pth')

    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    for i in range(2):
        batch = next(train_loader_iter)
        motion1, motion2, m_length1, m_length2, t1, t2 = batch
        input = {}
        input['motion_better'] = motion1
        input['motion_worse'] = motion2
        print(t1, t2)

        joints1 = recover_from_ric(motion1)

        output = net(input)
        print(output)


        save_path = os.path.join('./noisy_videos', )
        joints = recover_from_smpl(motion1)
        plot_3d_motion(save_path, skeleton, joints, dataset='humanml', title='111', fps=20)
        break












