import torch
import torch.nn.functional as F
import sys
from .metrics import *
from .model_util import get_clip_model
from .mask_utils import vis_motion
from os.path import join as pjoin

# @torch.no_grad()
# def evaluation_ADControl(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
#                                 time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
#                                               cal_mm=True, res_cond_scale=5):
@torch.no_grad()
def evaluation_ADControl(test_loader, eval_wrapper, diffusion_root, mean, std, args, logger, repeat_id, batch_size=32, diffusion=None, cal_mm=True):

    mean_for_eval = np.load('dataset/t2m_mean.npy')
    std_for_eval = np.load('dataset/t2m_std.npy')

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0

    clip_model = get_clip_model()

    for i, batch in enumerate(test_loader):
        # logger.info(f'{i}/{len(test_loader)}')
        print(f'{i}/{len(test_loader)}')
        word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask = batch
        b, max_length, num_features = gt_motion.shape
        gt_motion = gt_motion.cuda()
        real_length = real_length.cuda()
        traj = traj.cuda()
        traj_mask = traj_mask.cuda()
        traj_mask_263 = traj_mask_263.cuda()
        real_mask = generate_src_mask(max_length, real_length) # (b,196)
        gt_ric = gt_motion[..., :67]

        #encode text
        text = clip.tokenize(clip_text, truncate=True).cuda() 
        text_emb, word_emb = clip_model(text) # (b,512) 

        condition = {}
        condition['traj'] = traj
        condition['text_emb'] = text_emb
        condition['word_emb'] = word_emb
        condition['traj_mask'] = traj_mask
        condition['traj_mask_263'] = traj_mask_263
        condition['gt_motion'] = gt_motion
        condition['traj'] = traj
        condition['real_mask'] = real_mask
        condition['clip_text'] = clip_text

        if i < num_mm_batch:
            motion_multimodality_batch = []
            for _ in range(30):
                # 采样根节点轨迹
                if diffusion_root.modeltype == 'omni67': 
                    pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=condition, batch_size=batch_size)
                    # loss, msg = diffusion_root.calc_loss(gt_ric, pred_ric, mean[..., :67], std[..., :67], traj_mask, traj, real_mask, traj_mask_263, 67, 1)

                if args.normalize_traj:
                    traj = traj * diffusion_root.raw_std + diffusion_root.raw_mean

                ### 加上2阶段，但是只引导第一次，做Forced Guidance的消融实验
                if diffusion != None:
                    partial_emb = torch.zeros_like(gt_motion, device=gt_motion.device)
                    partial_emb[..., :67] = pred_ric # 使用dataloader出来的
                    # partial_emb[..., :67] = gt_ric 
                    pred_motion = diffusion.p_sample_loop(partial_emb, with_control=False, model_kwargs=condition, batch_size=batch_size, control_once=True)
                else:
                    pred_motion = pred_ric
                ##########

                et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, 
                                                                  pred_motion.clone(), real_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            # # 采样根节点轨迹
            if diffusion_root.modeltype == 'omni67': 
                pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=condition, batch_size=batch_size)
            #     # loss, msg = diffusion_root.calc_loss(gt_ric, pred_ric, mean[..., :67], std[..., :67], traj_mask, traj, real_mask, traj_mask_263, 67, 1)

            if args.normalize_traj:
                traj_denorm = traj * diffusion_root.raw_std + diffusion_root.raw_mean
            # recon_xyz = recover_from_ric(pred_ric[..., :67] * diffusion_root.std[..., :67] + diffusion_root.mean[..., :67], joints_num=22) 
            # loss_xyz_part = F.l1_loss(recon_xyz[traj_mask], traj_denorm[traj_mask]) # 仅约束控制轨迹
            # print('loss_xyz_part = ', loss_xyz_part)

            ### 加上2阶段，但是只引导第一次，做Forced Guidance的消融实验
            partial_emb = torch.zeros_like(gt_motion, device=gt_motion.device)
            partial_emb[..., :67] = pred_ric # 使用dataloader出来的
            # partial_emb[..., :67] = gt_ric ## debug 
            pred_motion = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=condition, batch_size=batch_size, control_once=False)
            # pred_motion = gt_motion
            ##########
            
            # if args.normalize_traj:
            #     traj = traj * diffusion_root.raw_std + diffusion_root.raw_mean
            # recon_xyz = recover_from_ric(pred_motion * diffusion_root.std + diffusion_root.mean, joints_num=22)
            # loss_xyz_part = F.l1_loss(recon_xyz[traj_mask], traj[traj_mask]) # 仅约束控制轨迹
            # print('loss_xyz_part = ', loss_xyz_part)

            # vis_motion(pred_motion, gt_motion)
            a = 1
                
            # 执行renorm
            normed_motion = motion.cpu().numpy()
            denormed_motion = test_loader.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - mean_for_eval) / std_for_eval  # according to T2M norms
            motion = torch.from_numpy(renormed_motion).cuda()


            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
                                                              pred_motion.clone(), real_length)
        

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, gt_motion, real_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += b

        # break

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy() # (4640,512)
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    # if sys.gettrace():
    #     diversity_real = 0
    #     diversity = 0
    # else:
    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    logger.info(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality