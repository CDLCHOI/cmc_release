
import sys
import options.option_transformer as option_trans
args = option_trans.get_args_parser()
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)
os.environ['OMP_NUM_THREADS'] = '8'
from utils.fixseed import fixseed
import torch
from data_loaders.humanml.utils.metrics import *
from datetime import datetime
import numpy as np
from collections import OrderedDict
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_control_dataset
from dataset import dataset_control
import warnings
warnings.filterwarnings('ignore')
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from utils.mask_utils import load_ckpt
from utils.motion_process import recover_from_ric, recover_from_rot, recover_root_rot_pos
from models.cfg_sampler import ClassifierFreeSampleModel

from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import cont6d_to_quat

def evaluate_control(motion_loaders, file):
    l2_dict = OrderedDict({})
    skating_ratio_dict = OrderedDict({})
    trajectory_score_dict = OrderedDict({})

    motion_loader_name = 'vald'
    motion_loader = motion_loaders[motion_loader_name]
    print('========== Evaluating Control ==========')
    # all_dist = []
    all_size = 0
    dist_sum = 0
    skate_ratio_sum = 0
    traj_err = []
    traj_err_key = traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
    # print(motion_loader_name)
    
    with torch.no_grad():
        for idx, batch in enumerate(motion_loader):
            word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, hint, filename = batch
            dim = motions.shape[-1]
            mean_for_eval = motion_loader.dataset.gen_loader.dataset.mean_for_eval[:dim]
            std_for_eval = motion_loader.dataset.gen_loader.dataset.std_for_eval[:dim]
            motions = motions * std_for_eval + mean_for_eval
            motions = motions.float()
            n_joints = 22 if motions.shape[-1] in [263, 67] else 21
            
            joints = recover_from_ric(motions, n_joints)
            if n_joints == 21:
                joints = joints * 0.001
            
            # foot skating error
            if n_joints == 21:
                skate_ratio, skate_vel = calculate_skating_ratio_kit(joints.permute(0, 2, 3, 1))  # [batch_size]
            else:
                skate_ratio, skate_vel = calculate_skating_ratio(joints.permute(0, 2, 3, 1))  # [batch_size]
            skate_ratio_sum += skate_ratio.sum()

            # control l2 error
            # process hint
            mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3).sum(dim=-1, keepdim=True) != 0
            raw_mean = motion_loader.dataset.gen_loader.dataset.t2m_dataset.raw_mean
            raw_std = motion_loader.dataset.gen_loader.dataset.t2m_dataset.raw_std
            hint = hint * raw_std + raw_mean
            if n_joints == 21:
                hint = hint * 0.001
            hint = hint.view(hint.shape[0], hint.shape[1], n_joints, 3) * mask_hint
            i = 0
            for motion, h, mask in zip(joints, hint, mask_hint):
                control_error = control_l2(motion.unsqueeze(0).numpy(), h.unsqueeze(0).numpy(), mask.unsqueeze(0).numpy())
                mean_error = control_error.sum() / mask.sum()
                dist_sum += mean_error
                control_error = control_error.reshape(-1)
                mask = mask.reshape(-1)
                err_np = calculate_trajectory_error(control_error, mean_error, mask)
                traj_err.append(err_np)
                # ferr.write(f'{filename[i]} {mean_error.item():.4f} {control_error.max():.4f}\n')
                i += 1

            all_size += joints.shape[0]

        # l2 dist
        dist_mean = dist_sum / all_size
        l2_dict[motion_loader_name] = dist_mean

        # Skating evaluation
        skating_score = skate_ratio_sum / all_size
        skating_ratio_dict[motion_loader_name] = skating_score

        ### For trajecotry evaluation from GMD ###
        traj_err = np.stack(traj_err).mean(0)
        trajectory_score_dict[motion_loader_name] = traj_err

    print(f'---> [{motion_loader_name}] Control L2 dist: {dist_mean:.4f}')
    print(f'---> [{motion_loader_name}] Control L2 dist: {dist_mean:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}')
    print(f'---> [{motion_loader_name}] Skating Ratio: {skating_score:.4f}', file=file, flush=True)
    line = f'---> [{motion_loader_name}] Trajectory Error: '
    for (k, v) in zip(traj_err_key, traj_err):
        line += '(%s): %.4f ' % (k, np.mean(v))
    print(line)
    print(line, file=file, flush=True)
    return l2_dict, skating_ratio_dict, trajectory_score_dict

def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    motiontmp = []
    txttmp = []
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        if motion_loader_name == 'ground truth':
            a = 1
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                if motion_loader_name == 'ground truth':
                    # data_control.py
                    word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _, _, _, _, filename = batch
                else:
                    # comp_v6_model_dataset.py
                    word_embeddings, pos_one_hots, caption, sent_lens, motions, m_lens, _, _, filename = batch

                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())


            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(eval_wrapper, groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _, _, _, _, filename= batch
            motion_embeddings = eval_wrapper.get_motion_embeddings(
                motions=motions,
                m_lens=m_lens
            )
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(eval_wrapper, mm_motion_loaders, file, mm_num_times):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times=1, diversity_times=300, mm_num_times=0, run_mm=False):
    with open(log_file, 'a+') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({}),
                                   'Control_l2': OrderedDict({}),
                                   'Skating Ratio': OrderedDict({}),
                                   'Trajectory Error': OrderedDict({})})

        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            control_l2_dict, skating_ratio_dict, trajectory_score_dict = evaluate_control(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(eval_wrapper, motion_loaders, f)


            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(eval_wrapper, gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times)

            if run_mm:
                print(f'Time: {datetime.now()}')
                print(f'Time: {datetime.now()}', file=f, flush=True)
                mm_score_dict = evaluate_multimodality(eval_wrapper, mm_motion_loaders, f, mm_num_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]
            if run_mm:
                for key, item in mm_score_dict.items():
                    if key not in all_metrics['MultiModality']:
                        all_metrics['MultiModality'][key] = [item]
                    else:
                        all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        mean_dict = {}
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)
            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                mean_dict[metric_name + '_' + model_name] = mean
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif metric_name == 'Trajectory Error':
                    traj_err_key = ["traj_fail_20cm", "traj_fail_50cm", "kps_fail_20cm", "kps_fail_50cm", "kps_mean_err(m)"]
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)): # zip(traj_err_key, mean):
                        line += '(%s): Mean: %.4f CInt: %.4f; ' % (traj_err_key[i], mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)
        return mean_dict



if __name__ == '__main__':
    from utils.model_util import create_gaussian_diffusion_simple, get_logger
    # fixseed(args.seed)
    
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    assert args.resume_trans is not None, 'Must specify resume_trans'
    assert args.S1_diffusion_step >0 and args.S2_diffusion_step >0
    
    if args.eval_mode == 'no_mm':
        num_samples_limit = args.max_samples
        run_mm = False
        mm_num_samples = 0 
        mm_num_repeats = 0 
        mm_num_times = 0 
        diversity_times = 300
        replication_times = args.replication_times 
    elif args.eval_mode == 'with_mm':
        num_samples_limit = args.max_samples
        run_mm = True
        mm_num_samples = 100 
        mm_num_repeats = 30 
        mm_num_times = 10 
        diversity_times = 300
        replication_times = args.replication_times 
    else:
        raise ValueError()


    if args.S1_diffusion_step < 1000 or args.S2_diffusion_step < 1000:
        args.use_ddim = 1

    if args.only_t2m_s2:
        log_file = f"{os.path.dirname(args.resume_trans)}/t2m"
    else:
        log_file = f"{os.path.dirname(args.resume_trans)}/joint_{str(args.control_joint).replace(' ','')}_density_{args.density}"

    log_file += f'_repeat{args.replication_times}'
    if args.S1_diffusion_step != 1000 or args.S2_diffusion_step != 1000:
        log_file += f'_{args.S1_diffusion_step}_{args.S2_diffusion_step}'
    
    log_file += f'_num{args.max_samples}'
    log_file += '.log'
    
    if sys.gettrace():
        log_file = f'output/debug/1.log'
    logger = get_logger('', file_path=log_file)
    logger.info(f'*************************************************************')
    logger.info("python " + " ".join(sys.argv))
    logger.info(f'log_file = {log_file}')
    logger.info(f'args.dataset_name = {args.dataset_name}')
    logger.info(f'args.resume_root = {args.resume_root}')
    logger.info(f'args.resume_trans = {args.resume_trans}')
    logger.info(f'control joint = {args.control_joint}, density = {args.density}')
    logger.info(f'args.guidance_param = {args.guidance_param}')
    logger.info(f'args.replication_times = {args.replication_times}')
    logger.info(f'args.S1_diffusion_step = {args.S1_diffusion_step}')
    logger.info(f'args.S2_diffusion_step = {args.S2_diffusion_step}')
    logger.info(f'args.only_t2m_s2 = {args.only_t2m_s2}')
    
    # stage1
    diffusion_root = None
    if args.resume_root is not None: 
        from models.omnimdm_spatial import CMDM
        from utils.model_util import get_s1_args
        net_root = CMDM(**get_s1_args(args, args.roottype))
        # try:
        #     load_ckpt(net_root, args.resume_root, key='trans')
        # except:
        load_ckpt(net_root, args.resume_root, key=None, strict=True)
        diffusion_root = create_gaussian_diffusion_simple(args, net_root, args.roottype)
        net_root.cuda()
        net_root.eval()

    # stage2
    from utils.model_util import get_s2_args
    from models.mdm import MDM
    net = MDM(**get_s2_args(args))
    load_ckpt(net, args.resume_trans, key=None, strict=True)
    net = ClassifierFreeSampleModel(net)
    diffusion = create_gaussian_diffusion_simple(args, net, args.modeltype)
    net.cuda()
    net.eval()
    

    gt_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='gt', split='test', shuffle=True, num_workers=0, drop_last=True)
    gen_loader = dataset_control.DataLoader(batch_size=args.batch_size, args=args, mode='eval', split='test', shuffle=True, num_workers=0, drop_last=True)
    eval_motion_loaders = {
        'vald': lambda: get_control_dataset(
            args, gen_loader, None, diffusion_root, diffusion, mm_num_samples, mm_num_repeats, num_samples_limit
        )
    }
    eval_wrapper = EvaluatorMDMWrapper(args.dataset_name, torch.device('cuda'))
    evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)
