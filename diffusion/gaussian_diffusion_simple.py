import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from eval_cmc import evaluation
from os.path import join as pjoin
from diffusion.respace import space_timesteps
from utils.motion_process import recover_from_ric
from diffusion.resample import create_named_schedule_sampler
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_control_dataset
from utils.mask_utils import root_dist_loss, generate_src_mask, vis_motion, calc_loss_xyz




class GaussianDiffusionSimple:
    def __init__(self, args, model, modeltype, clip_model, betas) -> None:
        self.args = args
        self.model = model
        self.modeltype = modeltype # 'ED'
        self.clip_model = clip_model
        self.eval_wrapper = EvaluatorMDMWrapper(args.dataset_name, torch.device('cuda'))
        self.gt_loader = None
        self.gen_loader = None
        self.log_file = None

        if self.args.dataset_name == 't2m':
            self.n_joints = 22
            self.mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy')).cuda()[None, None, ...] # dataset/HumanML3D/Mean.npy
            self.std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy')).cuda()[None, None, ...]
            self.raw_mean = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Mean_raw.npy')).cuda()[None, None, ...].view(1,1,22,3) 
            self.raw_std = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Std_raw.npy')).cuda()[None, None, ...].view(1,1,22,3)
        elif self.args.dataset_name == 'kit':
            self.n_joints = 21
            self.mean = torch.from_numpy(np.load('dataset/KIT-ML/Mean.npy')).cuda()[None, None, ...].float() # dataset/HumanML3D/Mean.npy
            self.std = torch.from_numpy(np.load('dataset/KIT-ML/Std.npy')).cuda()[None, None, ...].float()
            self.raw_mean = torch.from_numpy(np.load('dataset/kit_spatial_norm/Mean_raw.npy')).cuda()[None, None, ...].view(1,1,21,3) 
            self.raw_std = torch.from_numpy(np.load('dataset/kit_spatial_norm/Std_raw.npy')).cuda()[None, None, ...].view(1,1,21,3)

        

        # diffusion相关参数值
        betas = np.array(betas, dtype=np.float64) # 每个step的噪声方差，如果总共有T个step，那betas长度就是T
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0]) # 1000，若DDIM就是比如100

        # 这一部分用于前向加噪过程
        alphas = 1.0 - betas # (1000,)
        self.alphas_cumprod = np.cumprod(alphas, axis=0) # alpha t的累乘 # (1000,)
        
        #### DDIM 相关设定
        if args.use_ddim:
            print(' === initializing DDIM ')
            timestep_respacing = args.timestep_respacing
            assert timestep_respacing != '',"Subseq Undefined"

            self.use_timesteps = set(space_timesteps(self.num_timesteps, timestep_respacing))
            self.timestep_map = [] #for indexing timestep value from subseq index

            last_alpha_cumprod = 1.0
            new_betas = []
            for i, alpha_cumprod in enumerate(self.alphas_cumprod):
                if i in self.use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)

            self.betas = np.array(new_betas,dtype=np.float64)
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert (self.betas > 0).all() and (self.betas <= 1).all()

            self.timestep_map = torch.tensor(self.timestep_map,dtype=torch.int64).cuda()
            self.num_timesteps = int(self.betas.shape[0])
            alphas = 1.0 - self.betas # (1000,)
            self.alphas_cumprod = np.cumprod(alphas, axis=0)
        print(f"\n modeltype: {self.modeltype}, diffusion step: {self.num_timesteps} \n")
        #### DDIM 相关设定

        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1]) # alpha t-1的累乘 # (1000,)
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # 这一部分用于反向去噪过程
        # calculations for diffusion q(xt | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod) # DDPM原文公式（4）的x0系数  根号(alpha的累乘)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod) # 公式（3）的噪声系数
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | xt, x_0)
        
        # 2024-11-19 betas改self.betas 为支持DDIM
        self.posterior_variance = ( 
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) # 对应公式（7）中beta_t波浪
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # 2024-11-19 betas改self.betas 为支持DDIM
        self.posterior_mean_coef1 = ( 
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod) # 公式（7）中后验均值 x_0的系数
        )
        self.posterior_mean_coef2 = ( # 公式（7）中后验均值 x_t的系数
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
        self.schedule_sampler = create_named_schedule_sampler('uniform', self)

        self.predx0_list = []
        self.mean_list = []
        self.guide_list = []
        self.sample_list = []
        self.time_list = []


    def trainer_func_s1(self, dataloader_iter, logger, optimizer, scheduler, test_loader=None, dim=67):
        ''' train stage1 DiffRoot
        '''
        assert 'text' in self.model.cond_mode # add at 20240627
        min_err = 100
        for nb_iter in tqdm(range(1, self.args.total_iter+1), position=0, leave=True):
            batch = next(dataloader_iter)
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
            # clip_text, gt_token, m_tokens_len = batch
            gt_motion = gt_motion.cuda()
            gt_ric = gt_motion[..., :dim]
            b, max_length, num_features = gt_ric.shape
            real_length = real_length.cuda()
            traj = traj.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            traj_mask = traj_mask.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
            

            # text_emb = self.model.encode_text(clip_text) # 2024.10.10改

            condition = {}
            condition['clip_text'] = clip_text
            condition['traj'] = traj
            condition['traj_mask'] = traj_mask
            condition['traj_mask_263'] = traj_mask_263
            condition['gt_motion'] = gt_motion
            condition['traj'] = traj
            condition['real_mask'] = real_mask
            condition['real_length'] = real_length

            t, weights = self.schedule_sampler.sample(b, gt_ric.device) # timestep
            # t = torch.tensor([900]*b).cuda()
            x0 = gt_ric
            noise = torch.randn_like(x0) # 生成与x0形状一样的高斯噪声
            xt = self.q_sample(x0, t, noise=noise) # 给数据集x0加t步噪声
            if self.args.use_lbfgs:
                xt_tmp = self.lbfgs_guide(xt, t, condition)
            else:
                xt_tmp = self.guide(xt, t, condition, train=True) # spatial guidance

            xt_input = xt_tmp

            # 前向
            xt_input = xt_input.permute(0,2,1)[:,:,None]
            y={}
            y['text'] = clip_text
            y['hint'] = traj.flatten(2,3)

            pred_x0 = self.model(xt_input, t, y=y)
            pred_x0 = pred_x0.squeeze(2).permute(0,2,1)
            loss, msg = self.calc_loss(x0, pred_x0, self.mean, self.std, traj_mask, traj, real_mask, traj_mask_263, nb_iter)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if nb_iter % self.args.print_iter ==  0 :
                logger.info(msg)

            if nb_iter % self.args.save_iter == 0:
                torch.save(self.model.state_dict(), pjoin(self.args.out_dir, 'net_last.pth'))

    

    def trainer_func_s2(self, dataloader_iter, logger, optimizer, scheduler):
        for nb_iter in tqdm(range(1, self.args.total_iter+1), position=0, leave=True):
            batch = next(dataloader_iter)
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
            b, max_length, num_features = gt_motion.shape
            gt_motion = gt_motion.cuda()
            real_length = real_length.cuda()
            traj = traj.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            traj_mask = traj_mask.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)


            t, weights = self.schedule_sampler.sample(b, gt_motion.device) # timestep
            
            x0 = gt_motion

            # 加噪
            noise = torch.randn_like(x0) 
            xt = self.q_sample(x0, t, noise=noise) 
            
            # selective inpainting mechanism (SIM) 训练
            if self.args.sim:
                if np.random.choice([0,1]):
                    masked_xt = torch.where(traj_mask_263, x0, xt) # Forced Guidance
                else:
                    masked_xt = xt
            else:
                masked_xt = xt
                


            # 前向
            masked_xt = masked_xt.permute(0,2,1)[:,:,None]
            y={'text': clip_text}
            y['traj_mask_263'] = traj_mask_263.float()
            pred_x0 = self.model(masked_xt, t, y=y)  # (b,196,263)
            pred_x0 = pred_x0.squeeze(2).permute(0,2,1)
            
            loss, msg = self.calc_loss(x0, pred_x0, self.mean, self.std, traj_mask, traj, real_mask, traj_mask_263, nb_iter)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            
            if nb_iter % self.args.print_iter == 0 :
                logger.info(msg)

            if nb_iter % self.args.save_iter == 0:
                torch.save(self.model.state_dict(), pjoin(self.args.out_dir, 'net_last.pth'))

    def trainer_func_mdm(self, dataloader_iter, logger, optimizer, scheduler, writer):
        ''' 跑新idea的 不属于CMC '''
        for nb_iter in tqdm(range(1, self.args.total_iter+1), position=0, leave=True):
            batch = next(dataloader_iter)
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
            b, max_length, num_features = gt_motion.shape
            gt_motion = gt_motion.cuda()
            real_length = real_length.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)


            t, weights = self.schedule_sampler.sample(b, gt_motion.device) # timestep
            
            x0 = gt_motion

            # 加噪
            noise = torch.randn_like(x0) 
            xt = self.q_sample(x0, t, noise=noise) 
            masked_xt = xt

            # 前向
            masked_xt = masked_xt.permute(0,2,1)[:,:,None]
            y={'text': clip_text}
            y['traj_mask_263'] = traj_mask_263.float()
            pred_x0 = self.model(masked_xt, t, y=y)  # (b,196,263)
            pred_x0 = pred_x0.squeeze(2).permute(0,2,1)

            loss, msg = self.calc_mdm_loss(x0, pred_x0, real_length, real_mask, nb_iter, writer)
            
            # 检查 requires_grad 属性
            # for name, param in self.model.named_parameters():
            #     if 'clip' not in name:
            #         print(name, param.requires_grad, param.grad is None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # for name, param in self.model.named_parameters():
            #     if 'clip' not in name:
            #         print(name, param.requires_grad, param.grad is None)
            
            if nb_iter % self.args.print_iter == 0 :
                logger.info(msg)

            if nb_iter % self.args.save_iter == 0:
                torch.save(self.model.state_dict(), pjoin(self.args.out_dir, 'net_last.pth'))


    def calc_loss(self, gt, pred, mean, std, traj_mask, traj, real_mask, traj_mask_263, nb_iter, need_assert=True):
        loss = 0
        loss_xyz = 0
        loss_rotate_global = 0
        loss_position_global = 0

        
        dim = pred.shape[-1]

        # 损失函数计算
        motion_real_mask = real_mask[..., None].repeat(1,1, dim)

        # element-wise loss
        if self.args.loss_type == 'l1':
            loss_motion = F.l1_loss(pred[motion_real_mask], gt[motion_real_mask])
        elif self.args.loss_type == 'l2':
            loss_motion = F.mse_loss(pred[motion_real_mask], gt[motion_real_mask])
        else:
            raise NotImplementedError

        
        # 坐标loss
        # if self.args.normalize_traj:
        #     traj = traj * self.raw_std + self.raw_mean
        # recon_xyz = recover_from_ric(pred * std[..., :dim] + mean[..., :dim], joints_num=self.n_joints)  # 反归一化再转全局xyz
        # gt_xyz = recover_from_ric(gt * std[..., :dim] + mean[..., :dim], joints_num=self.n_joints)
        # # ipdb.set_trace()
        # if need_assert:
        #     if self.args.dataset_name == 't2m':
        #         assert torch.allclose(gt_xyz * traj_mask , traj * traj_mask, atol=1e-5) # 确保轨迹及mask是正确的
        #     elif self.args.dataset_name == 'kit':
        #         assert torch.allclose(gt_xyz * traj_mask / 1000, traj * traj_mask/ 1000, atol=1e-5) # 确保轨迹及mask是正确的

        # scale = 1 if self.args.dataset_name == 't2m' else 0.001
        # if self.args.loss_type == 'l1':
        #     loss_xyz = F.l1_loss(recon_xyz[traj_mask]*scale, traj[traj_mask]*scale) # 仅约束控制轨迹
        # elif self.args.loss_type == 'l2':
        #     loss_xyz = F.mse_loss(recon_xyz[traj_mask]*scale, traj[traj_mask]*scale) # 仅约束控制轨迹

        
        # 应该可以删掉 留着无所谓
        # if self.args.root_dist_loss:
        #     gt_root = (gt * std[..., :dim] + mean[..., :dim])[..., :4]
        #     recon_root = (pred * std[..., :dim] + mean[..., :dim])[..., :4]
        #     loss_rotate_global, loss_position_global, gt_root_pos, pred_root_pos = root_dist_loss(gt_root, recon_root, real_mask, self.args)
        #     loss += loss_rotate_global
        #     loss += loss_position_global
        # assert torch.allclose(gt_root_pos, gt_xyz[:,:,0,:])

            
        loss +=  loss_xyz
        loss = loss + loss_motion

        msg = f'Train. Iter {nb_iter} '
        msg += f" loss_motion. {loss_motion:.5f}, loss_xyz. {loss_xyz:.5f} "
        return loss, msg
    
    def calc_mdm_loss(self, gt, pred, real_length, real_mask, iter, writer):
        loss = 0
        msg = f' Train. Iter {iter} '

        dim = gt.shape[-1]
        motion_real_mask = real_mask[..., None].repeat(1,1, dim)

        loss_motion = F.mse_loss(pred[motion_real_mask], gt[motion_real_mask])
        msg += f" loss_motion. {loss_motion:.5f}"

        if not self.args.only_emb_loss:
            loss += loss_motion

        # with torch.no_grad():
        if self.args.emb_loss:
            gt_emb = self.eval_wrapper.get_motion_embeddings(
                    motions=gt,
                    m_lens=real_length
                )
                

            pred_emb = self.eval_wrapper.get_motion_embeddings(
                    motions=pred,
                    m_lens=real_length
                )
            
            emb_loss = F.mse_loss(gt_emb, pred_emb)
            loss += self.args.emb_loss * emb_loss
            msg += f" emb_loss. {emb_loss:.5f}"

        return loss, msg

    #############################################################################################################
    #############################################################################################################
    #############################################################################################################
    @torch.no_grad()
    def p_sample_loop(self, partial_emb, with_control=True, model_kwargs=None, batch_size=1, indices_bound=None):
        '''
        partial_emb: (b,196,263)
        condition: 字典，包含文本条件和轨迹条件
        '''
        if sys.gettrace():
            self.predx0_list = []
            self.mean_list = []
            self.guide_list = []

        B = batch_size # batch_size
        skip_t = 0
        if indices_bound is not None:
            assert isinstance(indices_bound, list)
            assert len(indices_bound) == 2 and indices_bound[0] > indices_bound[1]
            indices = list(range(indices_bound[0], indices_bound[1]-1, -1))
        else:
            indices = list(range(self.num_timesteps - skip_t))[::-1]

        if self.args.dataset_name == 't2m':
            motion_dim = 263
            ric_dim = 67
        elif self.args.dataset_name == 'kit':
            motion_dim = 251
            ric_dim = 64
        else:
            raise NotImplementedError

        if self.modeltype in ['diffmdm','mdm']:
            noise = torch.randn((B,196,motion_dim)).cuda()
        elif self.modeltype in ['omni67', 'omni67res', 'omni67mdm_spatial', 'semboost_67', 'mdm67_spatial']:
            noise = torch.randn((B,196,ric_dim)).cuda()
        else:
            print('self.modeltype = ', self.modeltype)
            raise ValueError('Unknown model type')

        xt = noise
        with torch.no_grad():
            if self.modeltype in ['diffmdm'] and partial_emb is not None:
                xt = torch.where(model_kwargs['traj_mask_263'], partial_emb, xt) 
            for i in tqdm(indices): # 999 ~ 0
                t = torch.tensor([i] * B).cuda() # timestep tensor
                if self.args.use_ddim:
                    out = self.ddim_sample(xt, t, partial_emb, model_kwargs=model_kwargs)
                else:
                    out = self.p_sample(xt, t, partial_emb, model_kwargs=model_kwargs) # 返回x_{t-1}和x0
                xt = out["sample"] # x_{t-1}
        
        # if sys.gettrace():
        #     self.plot_xyz_error(model_kwargs['traj'])

        # 只有CMC的2阶段会做替换
        if self.modeltype in ['diffmdm'] and with_control and partial_emb is not None: 
            out['sample'] = torch.where(model_kwargs['traj_mask_263'], partial_emb, out['sample']) 
        return out['sample']

    def p_sample(self, xt, t, partial_emb, model_kwargs=None):
        ''' get x_{t-1}
        '''
        B = xt.shape[0]
        out = self.p_mean_variance(xt, t, model_kwargs=model_kwargs) 

        '''
        一阶段对 mean 进行guide
        二阶段对 mean 进行替换
        '''
        if self.modeltype in ['omni67mdm_spatial']: # Spatial Guidance
            
            if self.args.use_lbfgs:
                # print('use bfgs')
                if self.args.bfgs_type == 0:
                    out['mean'] = self.lbfgs_guide(out['mean'], t, model_kwargs)
                elif self.args.bfgs_type == 1:
                    out['mean'] = self.lbfgs_guide1(out['mean'], t, model_kwargs)
                elif self.args.bfgs_type == 2:
                    out['mean'] = self.lbfgs_guide2(out['mean'], t, model_kwargs)
                else:
                    raise NotImplementedError

            else:
                out['mean'] = self.guide(out['mean'], t, condition=model_kwargs)
        elif self.modeltype in ['diffmdm']: # Forced Guidance
            # semboost用作1阶段纯文本生成时 partial_emb=None
            if partial_emb is not None: 
                out['mean'] = torch.where(model_kwargs['traj_mask_263'], partial_emb, out['mean']) # 将ric替换进去
        elif self.modeltype in ['mdm']:
            pass
        else:
            raise NotImplementedError

        mean = out['mean']   
        var = out['variance']
        log_var = out['log_variance']
        pred_x0 = out['pred_x0']

        # if sys.gettrace():
        #     self.guide_list.append(mean)

        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1))) # no noise when t == 0
        sample_all_noisy = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise # 全部加噪的认为是无条件
        # 此处去掉了return_type为priorMDM的代码，即mask为1的位置，将noise置0，认为mask为1的位置的值视为条件，仅当return_type为priorMDM时，sample_all_noisy和sample才有区别，不然是一样的，所以sample_all_noisy其实也没啥用
        sample = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise # noise是标准正态分布，sample是通过预测噪声，计算得到方差，再通过重参数化采样得到的x_{t-1}； 替换值不加噪的认为是有条件
            

        return {"sample": sample,
                'sample_all_noisy': sample_all_noisy, 
                "pred_x0": pred_x0} # 分别是x_{t-1}和x_0
    
    def ddim_sample(self, xt, t, partial_emb, model_kwargs=None, eta=0.5):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        B = xt.shape[0]
        out = self.p_mean_variance(xt, t, model_kwargs=model_kwargs)      

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(xt, t, out["pred_x0"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, xt.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, xt.shape)
        sigma = (eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        # Equation 12.
        noise = torch.randn_like(xt)
        mean_pred = (
            out["pred_x0"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )


        if self.modeltype in ['omni67mdm_spatial']: # Spatial Guidance
            if self.args.use_lbfgs:
                # print('use bfgs')
                if self.args.bfgs_type == 0:
                    mean_pred = self.lbfgs_guide(mean_pred, t, model_kwargs)
                elif self.args.bfgs_type == 1:
                    mean_pred = self.lbfgs_guide1(mean_pred, t, model_kwargs)
                elif self.args.bfgs_type == 2:
                    mean_pred = self.lbfgs_guide2(mean_pred, t, model_kwargs)
                else:
                    raise NotImplementedError
            else:
                mean_pred = self.guide(mean_pred, t, condition=model_kwargs)
        if self.modeltype in ['semboost','diffmdm']: # Forced Guidance
            mean_pred = torch.where(model_kwargs['traj_mask_263'], partial_emb, mean_pred) 


        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise


        return {"sample": sample,
                "pred_x0": out['pred_x0']}


    def p_mean_variance(self, masked_xt, t, model_kwargs=None):
        ''' get pred_x0
        '''
        B = masked_xt.shape[0]
        assert t.shape == (B,)
        
        traj = model_kwargs['traj']
        clip_text = model_kwargs['clip_text']
        
        assert masked_xt.shape[0] == len(clip_text) == traj.shape[0]
        if self.args.use_ddim:   # 2024-11-20添加，用于DDIM
            sample_t = self.timestep_map[t]
        else:
            sample_t = t.clone()
        # masked_xt = torch.ones_like(masked_xt, device=masked_xt.device) * 0.5; print(' for debug')
        # 前向推理
        a = time.time()
        if self.modeltype in ['omni67mdm_spatial']:
            xt = masked_xt.permute(0,2,1)[:,:,None]
            # 在这里给固定噪声看看
            # y = {'text':clip_text,  'text_emb':model_kwargs['text_emb']}
            y = {'text':clip_text}
            if y.get('text_emb') is not None:
                y['text_emb'] = y.get('text_emb')

            y['hint'] = traj.flatten(2,3)
            # scale = torch.ones(B,device=torch.device('cuda')) * 2.5 # 引导系数
            scale = torch.ones(B,device=torch.device('cuda')) * self.args.guidance_param
            y['scale'] = scale
            if self.modeltype == 'omni67res':
                y['traj_mask_67'] = model_kwargs['traj_mask_263'][..., :67].permute(0,2,1)[:,:,None]
            pred_x0 = self.model(xt, sample_t, y=y)
            pred_x0 = pred_x0.squeeze(2).permute(0,2,1)
        elif self.modeltype in ['diffmdm', 'mdm']:
            xt = masked_xt.permute(0,2,1)[:,:,None]
            # scale = torch.ones(B,device=torch.device('cuda')) * 2.5 # 引导系数
            scale = torch.ones(B,device=torch.device('cuda')) * self.args.guidance_param
            y={'text': clip_text, 'scale':scale}
            if y.get('text_emb') is not None:
                y['text_emb'] = y.get('text_emb')
            y['traj_mask_263'] = model_kwargs['traj_mask_263'].float()
            pred_x0 = self.model(xt, sample_t, y=y)  # (b,196,263)
            pred_x0 = pred_x0.squeeze(2).permute(0,2,1)
        else:
            raise NotImplementedError
        b = time.time()
        self.time_list.append(b-a)
        
        # if sys.gettrace():
        #     self.predx0_list.append(pred_x0)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        model_variance = _extract_into_tensor(model_variance, t, masked_xt.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, masked_xt.shape)

        # 得到x0后去算x_{t-1}的均值，即后验均值 q(x_{t-1} | x_t, x_0)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_x0, x_t=masked_xt, t=t) 
        # if sys.gettrace():
        #     self.mean_list.append(model_mean)

        assert model_mean.shape == model_log_variance.shape == pred_x0.shape == masked_xt.shape
        
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x0": pred_x0,
        }
    
    
    

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        assert noise.shape == x0.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
            * noise
        )
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        DDPM原论文公式(7) q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = ( # 公式（7）中的mu_t就是这个后验均值
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def masked_joint_loss(self, x, mean, std, hint, mask_hint):
        dim = x.shape[-1]
        x_ = x * std[..., :dim] + mean[..., :dim]
        n_joints = 22 if dim in [67, 193, 259, 263] else 21

        joint_pos = recover_from_ric(x_, n_joints) # 全局xyz
        if n_joints == 21: # KIT格式, 把毫米转为米
            joint_pos = joint_pos * 0.001
            hint = hint * 0.001

        loss = torch.norm((joint_pos - hint) * mask_hint, dim=-1)
        return loss

    def lbfgs_guide(self, x, t, condition, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        
        sample_t = t # self.timestep_map[t] if self.args.use_ddim_sample else t
        n_joint = 22 if x.shape[-1] in [67, 193, 259, 263] else 21
        threshold = 10 # self.timestep_map[1] if self.args.use_ddim_sample else 10
        
        if train:
            if sample_t[0] < threshold:
                n_guide_steps = 10
            else:
                n_guide_steps = 1
        else:
            if sample_t[0] < threshold:
                n_guide_steps = 10
            else:
                n_guide_steps = 1
        
        mask_hint = condition['traj_mask']
        # mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        hint = condition['traj'].clone().detach()
        
        if self.args.normalize_traj:
        # process hint
            if self.raw_std.device != hint.device:
                self.raw_mean = self.raw_mean.to(hint.device)
                self.raw_std = self.raw_std.to(hint.device)
                self.mean = self.mean.to(hint.device)
                self.std = self.std.to(hint.device)
            # 判断是否外部给定了mean和std，在测试集时候使用
            mean = condition.get('mean', None)
            std = condition.get('std', None)
            if mean is None and std is None:
                mean = self.mean
                std = self.std
            hint = hint * self.raw_std + self.raw_mean
            hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint  

        with torch.enable_grad():
            x = x.clone().detach().contiguous().requires_grad_(True)

            def closure():
                lbfgs.zero_grad()
                loss = self.masked_joint_loss(x, self.mean, self.std, hint, mask_hint).sum()
                loss.backward()
                return loss

            if self.modeltype == 'omni67res':
                lbfgs = torch.optim.LBFGS([x],
                        history_size=10, 
                        max_iter=1,
                        lr = 1.0,
                        tolerance_grad=1e-6,
                        line_search_fn="strong_wolfe")
            else:
                lbfgs = torch.optim.LBFGS([x],
                        history_size=10, 
                        max_iter=10,
                        lr = self.args.bfgs_lr,
                        tolerance_grad=1e-6,
                        line_search_fn="strong_wolfe")
            if t[0] >= t_stopgrad:
                for _ in range(n_guide_steps):
                    lbfgs.step(closure)
        return x

    def lbfgs_guide1(self, x, t, condition, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        
        sample_t = t # self.timestep_map[t] if self.args.use_ddim_sample else t
        n_joint = 22 if x.shape[-1] in [67, 193, 259, 263] else 21
        threshold = 10 # self.timestep_map[1] if self.args.use_ddim_sample else 10
        
        if train:
            if sample_t[0] < threshold:
                n_guide_steps = 10
            else:
                n_guide_steps = 1
        else:
            # 20241107修改后的bfgstype_1，优化部署仅包含10和100步，和bfgs2区别是最后一个step的1000步改为100步，做对照
            if sample_t[0] < threshold:  
                n_guide_steps = 10 # 0 ~ 9
            else:
                n_guide_steps = 1 # 10 ~ 999

            # if sample_t[0] >= 10:
            #     n_guide_steps = 1 # 10 ~ 999
            # elif sample_t[0] < 10 and sample_t[0] >0:
            #     n_guide_steps = 10  # 1 ~ 9
            # else:
            #     n_guide_steps = 100 # 0

        lrs = np.linspace(self.args.bfgs_lr,0.1,1000)
        lr = lrs[sample_t[0]]
        
        mask_hint = condition['traj_mask']
        # mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        hint = condition['traj'].clone().detach()
        
        if self.args.normalize_traj:
        # process hint
            if self.raw_std.device != hint.device:
                self.raw_mean = self.raw_mean.to(hint.device)
                self.raw_std = self.raw_std.to(hint.device)
                self.mean = self.mean.to(hint.device)
                self.std = self.std.to(hint.device)
            # 判断是否外部给定了mean和std，在测试集时候使用
            mean = condition.get('mean', None)
            std = condition.get('std', None)
            if mean is None and std is None:
                mean = self.mean
                std = self.std
            hint = hint * self.raw_std + self.raw_mean
            hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint  

        with torch.enable_grad():
            x = x.clone().detach().contiguous().requires_grad_(True)

            def closure():
                lbfgs.zero_grad()
                # loss = self.masked_joint_loss(x, self.mean, self.std, hint, mask_hint).sum()
                loss = self.masked_joint_loss(x, self.mean, self.std, hint, mask_hint).mean()
                loss.backward()
                return loss
            lbfgs = torch.optim.LBFGS([x],
                        history_size=10, 
                        max_iter=10*n_guide_steps,
                        lr = lr,
                        # tolerance_grad=1e-6,
                        tolerance_change=1e-8,
                        line_search_fn="strong_wolfe")
                
            lbfgs.step(closure)
        return x
    
    
    def lbfgs_guide2(self, x, t, condition):
        
        sample_t = t # self.timestep_map[t] if self.args.use_ddim_sample else t
        n_joint = 22 if x.shape[-1] in [67, 193, 259, 263] else 21
        
        
        if sample_t[0] >= 10:
            n_guide_steps = 1 # 10 ~ 999
        elif sample_t[0] < 10 and sample_t[0] >0:
            n_guide_steps = 10  # 1 ~ 9
        else:
            n_guide_steps = 100 # 0

        lrs = np.linspace(self.args.bfgs_lr,0.1,1000)
        lr = lrs[sample_t[0]]
        
        mask_hint = condition['traj_mask']
        # mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        hint = condition['traj'].clone().detach()
        
        if self.args.normalize_traj:
        # process hint
            if self.raw_std.device != hint.device:
                self.raw_mean = self.raw_mean.to(hint.device)
                self.raw_std = self.raw_std.to(hint.device)
                self.mean = self.mean.to(hint.device)
                self.std = self.std.to(hint.device)
            # 判断是否外部给定了mean和std，在测试集时候使用
            mean = condition.get('mean', None)
            std = condition.get('std', None)
            if mean is None and std is None:
                mean = self.mean
                std = self.std
            hint = hint * self.raw_std + self.raw_mean
            hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint  

        with torch.enable_grad():
            x = x.clone().detach().contiguous().requires_grad_(True)

            def closure():
                lbfgs.zero_grad()
                # loss = self.masked_joint_loss(x, self.mean, self.std, hint, mask_hint).sum()
                loss = self.masked_joint_loss(x, self.mean, self.std, hint, mask_hint).mean()
                loss.backward()
                return loss

            lbfgs = torch.optim.LBFGS([x],
                        history_size=10, 
                        max_iter=10*n_guide_steps,
                        lr = lr,
                        tolerance_change=1e-8,
                        line_search_fn="strong_wolfe")
                
            lbfgs.step(closure)
        return x
    

    def guide(self, x, t, condition, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        sample_t = self.timestep_map[t] if self.args.use_ddim else t
        n_joint = 22 if x.shape[-1] in [67, 193, 259, 263] else 21
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        model_variance = torch.exp(model_log_variance)
        
        if model_variance[0, 0, 0] < min_variance:
            model_variance = min_variance

        if train:
            if t[0] < 20:
                n_guide_steps = 100
            else:
                n_guide_steps = 20
        else:
            if t[0] < 10:
                n_guide_steps = 500
            else:
                n_guide_steps = 10
            # if self.modeltype == 'omni67res':
            #     n_guide_steps = 5
        

        mask_hint = condition['traj_mask']
        # mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        hint = condition['traj'].clone().detach()
        if self.args.normalize_traj:
            # process hint
            if self.raw_std.device != hint.device:
                self.raw_mean = self.raw_mean.to(hint.device)
                self.raw_std = self.raw_std.to(hint.device)
                self.mean = self.mean.to(hint.device)
                self.std = self.std.to(hint.device)
            # 判断是否外部给定了mean和std，在测试集时候使用
            mean = condition.get('mean', None)
            std = condition.get('std', None)
            if mean is None and std is None:
                mean = self.mean
                std = self.std
            hint = hint * self.raw_std + self.raw_mean
            hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint

        
        if not train:
            scale = self.calc_grad_scale(mask_hint[..., :1]) * self.sgd_weight # omnicontrol这里的mask输入shape是 (b,196,22,1)
            # if self.modeltype == 'omni67res':
            #     scale = torch.ones_like(scale, device=scale.device) * 1
            # a = torch.linspace(1, 3, steps=196).to(mask_hint.device)
            # weight = (a**1)[None, :, None]
            # scale = scale * 3

        for _ in range(n_guide_steps):
            loss, grad = self.gradients(x, self.mean, self.std, hint, mask_hint, condition['real_length']) # x和hint都是未归一化的
            # if t[0] == 0:
                # print('loss.sum() = ',loss.sum())
            grad = model_variance * grad
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        return x.detach()
    
    def calc_grad_scale(self, mask_hint):
        assert mask_hint.shape[1] == 196
        num_keyframes = mask_hint.sum(dim=1).squeeze(-1)
        max_keyframes = num_keyframes.max(dim=1)[0]
        scale = 20 / max_keyframes
        if self.modeltype in ['omni67', 'omni193', 'omni259', 'omnicontrol', 'omni67mdm_spatial', 'omni263mdm_spatial',
                              'omni67res', 'omni263mdm_fuse']:
            return scale.unsqueeze(-1).unsqueeze(-1)
        else:
            raise NotImplementedError

    def gradients(self, x, mean, std, hint, mask_hint, real_length, joint_ids=None):
        with torch.enable_grad():
            x.requires_grad_(True)
            b = x.shape[0]
            dim = x.shape[-1]
            x_ = x * std[..., :dim] + mean[..., :dim]
            n_joints = 22 if dim in [67, 193, 259, 263] else 21
            joint_pos = recover_from_ric(x_, n_joints) # 全局xyz (b,196,22,3)
            if n_joints == 21: # 猜测是KIT格式, 就要把 毫米转为米？
                joint_pos = joint_pos * 0.001
                hint = hint * 0.001

            loss = torch.norm((joint_pos - hint) * mask_hint, dim=-1)
            grad = torch.autograd.grad([loss.sum()], [x])[0] # （b, l, 67）
            # the motion in HumanML3D always starts at the origin (0,y,0), so we zero out the gradients for the root joint
            # plt.clf(); plt.ylim(-0.1,0.5); plt.plot(grad[0,:,0].cpu().numpy()); plt.title('root orientation'); plt.savefig('root_orientation.png')
            # plt.clf(); plt.ylim(-0.1,0.5); plt.plot(grad[0,:,1].cpu().numpy()); plt.title('root x'); plt.savefig('root_x.png')
            # plt.clf(); plt.ylim(-0.2,0.5); plt.plot(grad[0,:,2].cpu().numpy()); plt.title('root z'); plt.savefig('root_z.png')
            # plt.clf(); plt.ylim(-0.2,0.5); plt.plot(grad[0,:,3].cpu().numpy()); plt.title('root y'); plt.savefig('root_y.png')
            # grad[:, 0, :] = 0 # 第0帧梯度置0
            # ##### 梯度误差修正
            # control_batch_id = mask_hint.sum(1).sum(-1).nonzero() # (b, 2)
            # cb = control_batch_id[:,0]
            # ci = control_batch_id[:,1]
            # err = torch.norm(((joint_pos - hint) * mask_hint)[cb,:,ci,:], dim=-1, keepdim=True) # (b,196,1)
            # grad *= err # 0.61

            
            # coef_one = torch.ones((b,196,1)).cuda()
            ## 第1种
            # coef = torch.arange(real_length.item()).float().cuda() - real_length//2
            # coef += (coef==0).float()
            # coef = coef.abs() ** coef.sign()
            # coef= torch.cat([coef[None,:,None], torch.zeros([1,196-60,1]).cuda()], dim=1)
            ## 第2种
            # coef = torch.arange(real_length.item()).float().cuda() - real_length//2
            # coef += (coef==0).float()
            # coef[coef<0] = 1 # 一般控制不佳都是发生在后半段所以这么试
            # coef= torch.cat([coef[None,:,None], torch.zeros([1,196-60,1]).cuda()], dim=1)
            ### batch操作
            # for i in range(b):
            #     coef = torch.arange(real_length[i].item()).float().cuda() - real_length[i].item()//2
            #     coef += (coef==0).float()
            #     coef[coef<0] = 1 # 一般控制不佳都是发生在后半段所以这么试
            #     coef= torch.cat([coef[:,None], torch.zeros([196-real_length[i].item(),1]).cuda()], dim=0)
            #     coef_zero[i] = coef
            ## 第3种，直接按帧来，第0帧除以196，最后一帧不变

            # grad *= coef_one
            # grad[:,:,[1,2]] *= coef
            # grad[:,:,[1,2]] *= err

            # if self.args.root_zero_grad:
            #     grad[:, :, :4] = 0
            x.detach()
        return loss, grad
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    
    def plot_xyz_error(self, traj):
        mask_hint = traj.view(traj.shape[0], traj.shape[1], 22, 3).sum(dim=-1, keepdim=True) != 0  
        mean = self.mean[..., :67] if '67' in self.modeltype else self.mean
        std = self.std[..., :67] if '67' in self.modeltype else self.std
        if self.args.normalize_traj:
            traj = traj * self.raw_std + self.raw_mean

        b = traj.shape[0]
        if b==1:
            s = 0
        else:
            s = 1
        idx=torch.tensor([s + b*i for i in range(1000)]).cuda()
        traj = traj[s:s+1]
        mask_hint = mask_hint[s:s+1]
        x0s = torch.cat(self.predx0_list, dim=0)[idx] * std + mean
        means = torch.cat(self.mean_list, dim=0)[idx] * std + mean
        guides = torch.cat(self.guide_list, dim=0)[idx] * std + mean
        x0s_xyz = recover_from_ric(x0s, 22)
        means_xyz = recover_from_ric(means, 22)
        guides_xyz = recover_from_ric(guides, 22)
        # traj = traj.repeat(1000,1,1,1)
        # traj_mask = traj_mask.repeat(1000,1,1,1)
        
        # x0_err = F.l1_loss(x0s_xyz[traj_mask], traj[traj_mask])
        # mean_err = F.l1_loss(means_xyz[traj_mask], traj[traj_mask])
        # guide_err = F.l1_loss(guides_xyz[traj_mask], traj[traj_mask])
        from utils.metrics import control_l2
        x0_err = []
        mean_err = []
        guide_err = []
        batch = x0s.shape[0]//self.num_timesteps
        for i in range(self.num_timesteps):
            a = control_l2(x0s_xyz[i:(i+1)].cpu().numpy(), traj.cpu().numpy(), mask_hint.cpu().numpy()).sum() / mask_hint.sum() 
            b = control_l2(means_xyz[i:(i+1)].cpu().numpy(), traj.cpu().numpy(), mask_hint.cpu().numpy()).sum() / mask_hint.sum() 
            c = control_l2(guides_xyz[i:(i+1)].cpu().numpy(), traj.cpu().numpy(), mask_hint.cpu().numpy()).sum() / mask_hint.sum() 
            x0_err.append(a.item())
            mean_err.append(b.item())
            guide_err.append(c.item())

        
        x = np.arange(self.num_timesteps)
        x0_err = np.array(x0_err)
        mean_err = np.array(mean_err)
        guide_err = np.array(guide_err)
        print('pred_x0 err = ', a.item())
        # print('b = ', b.item())
        print('guide err = ', c.item())
        print('min guide err = ', guide_err.min())
        print('time cost = ', np.array(self.time_list).mean())

        
        plt.plot(x, x0_err, color='r')
        plt.plot(x, mean_err, color='g')
        plt.plot(x, guide_err, color='b')
        plt.savefig('0_1000.png')

        # if os.path.exists('x0_err_s1.npy'):
        #     np.save('x0_err_s2.npy', x0_err)
        #     np.save('mean_err_s2.npy', mean_err)
        #     np.save('guide_err_s2.npy',guide_err)
        # else:

        np.save('x0_err_s1.npy', x0_err)
        np.save('mean_err_s1.npy', mean_err)
        np.save('guide_err_s1.npy',guide_err)

        del self.predx0_list[:]
        del self.mean_list[:]
        del self.guide_list[:]

    # def plot_xyz_error(self, traj):
    #     b = traj.shape[0]
    #     mask_hint = traj.view(traj.shape[0], traj.shape[1], 22, 3).sum(dim=-1, keepdim=True) != 0  
    #     mean = self.mean[..., :67] if '67' in self.modeltype else self.mean
    #     std = self.std[..., :67] if '67' in self.modeltype else self.std
    #     if self.args.normalize_traj:
    #         traj = traj * self.raw_std + self.raw_mean

    #     s = 0  # 选择哪个数据
    #     idx=torch.linspace(0,b*1000,1001).int() + s
    #     x0s = torch.cat(self.predx0_list, dim=0)[idx[:1000]] * std + mean # (1,196,67)
    #     means = torch.cat(self.mean_list, dim=0)[idx] * std + mean
    #     guides = torch.cat(self.guide_list, dim=0)[idx[:1000]] * std + mean
    #     x0s_xyz = recover_from_ric(x0s, 22) # (1,196,22,3)
    #     means_xyz = recover_from_ric(means, 22)
    #     guides_xyz = recover_from_ric(guides, 22)
    #     # traj = traj.repeat(1000,1,1,1)
    #     # traj_mask = traj_mask.repeat(1000,1,1,1)
        
    #     # x0_err = F.l1_loss(x0s_xyz[traj_mask], traj[traj_mask])
    #     # mean_err = F.l1_loss(means_xyz[traj_mask], traj[traj_mask])
    #     # guide_err = F.l1_loss(guides_xyz[traj_mask], traj[traj_mask])
    #     from utils.metrics import control_l2
    #     x0_err = []
    #     mean_err = []
    #     guide_err = []
    #     b = x0s.shape[0]//self.num_timesteps
    #     for i in range(self.num_timesteps):
    #         a = control_l2(x0s_xyz[i:i+1].cpu().numpy(), traj[s:s+1].cpu().numpy(), mask_hint[s:s+1].cpu().numpy()).sum() / mask_hint[s:s+1].sum() 
    #         b = control_l2(means_xyz[b*i:b*(i+1)].cpu().numpy(), traj.cpu().numpy(), mask_hint.cpu().numpy()).sum() / mask_hint.sum() 
    #         c = control_l2(guides_xyz[i:i+1].cpu().numpy(), traj[s:s+1].cpu().numpy(), mask_hint[s:s+1].cpu().numpy()).sum() / mask_hint[s:s+1].sum() 
    #         x0_err.append(a.item())
    #         # mean_err.append(b.item())
    #         guide_err.append(c.item())

        
    #     x = np.arange(self.num_timesteps)
    #     x0_err = np.array(x0_err)
    #     # mean_err = np.array(mean_err)
    #     guide_err = np.array(guide_err)
    #     print('pred_x0 err = ', a.item())
    #     # print('posterior err = ', b.item())
    #     print('guide err = ', c.item())
    #     print('min guide err = ', guide_err.min())
    #     print('time cost = ', np.array(self.time_list).mean())

        
    #     plt.plot(x, x0_err, color='r')
    #     # plt.plot(x, mean_err, color='g')
    #     plt.plot(x, guide_err, color='b')
    #     plt.savefig('0_1000.png')

    #     # if os.path.exists('x0_err_s1.npy'):
    #     #     np.save('x0_err_s2.npy', x0_err)
    #     #     np.save('mean_err_s2.npy', mean_err)
    #     #     np.save('guide_err_s2.npy',guide_err)
    #     # else:

    #     np.save('x0_err_s1.npy', x0_err)
    #     # np.save('mean_err_s1.npy', mean_err)
    #     np.save('guide_err_s1.npy',guide_err)





def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

