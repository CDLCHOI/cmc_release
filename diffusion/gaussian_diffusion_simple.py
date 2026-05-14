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
        self.modeltype = modeltype
        self.clip_model = clip_model
        self.eval_wrapper = EvaluatorMDMWrapper(args.dataset_name, torch.device('cuda'))
        self.gt_loader = None
        self.gen_loader = None
        self.log_file = None

        if self.args.dataset_name == 't2m':
            self.n_joints = 22
            self.mean = torch.from_numpy(np.load('dataset/HumanML3D/Mean.npy')).cuda()[None, None, ...]
            self.std = torch.from_numpy(np.load('dataset/HumanML3D/Std.npy')).cuda()[None, None, ...]
            self.raw_mean = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Mean_raw.npy')).cuda()[None, None, ...].view(1,1,22,3) 
            self.raw_std = torch.from_numpy(np.load('dataset/humanml_spatial_norm/Std_raw.npy')).cuda()[None, None, ...].view(1,1,22,3)
        elif self.args.dataset_name == 'kit':
            self.n_joints = 21
            self.mean = torch.from_numpy(np.load('dataset/KIT-ML/Mean.npy')).cuda()[None, None, ...].float()
            self.std = torch.from_numpy(np.load('dataset/KIT-ML/Std.npy')).cuda()[None, None, ...].float()
            self.raw_mean = torch.from_numpy(np.load('dataset/kit_spatial_norm/Mean_raw.npy')).cuda()[None, None, ...].view(1,1,21,3) 
            self.raw_std = torch.from_numpy(np.load('dataset/kit_spatial_norm/Std_raw.npy')).cuda()[None, None, ...].view(1,1,21,3)

        

        # diffusion parameters
        betas = np.array(betas, dtype=np.float64)
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


    def trainer_func_s1(self, dataloader_iter, logger, optimizer, scheduler):
        ''' train stage1 DiffRoot
        '''
        if self.args.dataset_name == 't2m':
            dim = 67
        elif self.args.dataset_name == 'kit':
            dim = 64
        assert 'text' in self.model.cond_mode
        for nb_iter in tqdm(range(1, self.args.total_iter+1), position=0, leave=True):
            batch = next(dataloader_iter)
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
            gt_motion = gt_motion.cuda()
            gt_ric = gt_motion[..., :dim]
            b, max_length, num_features = gt_ric.shape
            real_length = real_length.cuda()
            traj = traj.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            traj_mask = traj_mask.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
            

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
            noise = torch.randn_like(x0)
            xt = self.q_sample(x0, t, noise=noise)
            xt_tmp = self.lbfgs_guide2(xt, t, condition)

            xt_input = xt_tmp

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
            
            # selective inpainting mechanism (SIM)
            if np.random.choice([0,1]):
                masked_xt = torch.where(traj_mask_263, x0, xt) # Forced Guidance
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


    def calc_loss(self, gt, pred, mean, std, traj_mask, traj, real_mask, traj_mask_263, nb_iter, need_assert=True):
        loss = 0
        loss_xyz = 0
        loss_rotate_global = 0
        loss_position_global = 0
        dim = pred.shape[-1]

        motion_real_mask = real_mask[..., None].repeat(1,1, dim)

        # loss_motion
        if self.args.loss_type == 'l1':
            loss_motion = F.l1_loss(pred[motion_real_mask], gt[motion_real_mask])
        elif self.args.loss_type == 'l2':
            loss_motion = F.mse_loss(pred[motion_real_mask], gt[motion_real_mask])
        else:
            raise NotImplementedError
        
        traj = traj * self.raw_std + self.raw_mean
        recon_xyz = recover_from_ric(pred * std[..., :dim] + mean[..., :dim], joints_num=self.n_joints)
        gt_xyz = recover_from_ric(gt * std[..., :dim] + mean[..., :dim], joints_num=self.n_joints)
        
        if need_assert:  # 确保轨迹及mask是正确的
            if self.args.dataset_name == 't2m':
                assert torch.allclose(gt_xyz * traj_mask , traj * traj_mask, atol=1e-5)
            elif self.args.dataset_name == 'kit':
                assert torch.allclose(gt_xyz * traj_mask / 1000, traj * traj_mask/ 1000, atol=1e-5)

        # loss_xyz
        scale = 1 if self.args.dataset_name == 't2m' else 0.001
        if self.args.loss_type == 'l1':
            loss_xyz = F.l1_loss(recon_xyz[traj_mask]*scale, traj[traj_mask]*scale)
        elif self.args.loss_type == 'l2':
            loss_xyz = F.mse_loss(recon_xyz[traj_mask]*scale, traj[traj_mask]*scale)

        # loss_root
        gt_root = (gt * std[..., :dim] + mean[..., :dim])[..., :4]
        recon_root = (pred * std[..., :dim] + mean[..., :dim])[..., :4]
        loss_rotate_global, loss_position_global, gt_root_pos, pred_root_pos = root_dist_loss(gt_root, recon_root, real_mask, self.args)
        loss += loss_rotate_global
        loss += loss_position_global
        assert torch.allclose(gt_root_pos, gt_xyz[:,:,0,:])

        loss +=  loss_xyz
        loss = loss + loss_motion
        msg = f'Train. Iter {nb_iter} '
        msg += f" loss_motion. {loss_motion:.5f}, loss_xyz. {loss_xyz:.5f} "
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

        if self.modeltype == 's1':
            noise = torch.randn((B,196,ric_dim)).cuda()
        elif self.modeltype == 's2':
            noise = torch.randn((B,196,motion_dim)).cuda()
        else:
            raise ValueError(f'Unknown model type {self.modeltype}')

        xt = noise
        with torch.no_grad():
            if self.modeltype == 's2' and partial_emb is not None:
                xt = torch.where(model_kwargs['traj_mask_263'], partial_emb, xt) 
            for i in tqdm(indices): # 999 ~ 0
                t = torch.tensor([i] * B).cuda() # timestep tensor
                if self.args.use_ddim:
                    out = self.ddim_sample(xt, t, partial_emb, model_kwargs=model_kwargs)
                else:
                    out = self.p_sample(xt, t, partial_emb, model_kwargs=model_kwargs)
                xt = out["sample"]
        
        if self.modeltype == 's2' and partial_emb is not None: 
            out['sample'] = torch.where(model_kwargs['traj_mask_263'], partial_emb, out['sample']) 
        return out['sample']

    def p_sample(self, xt, t, partial_emb, model_kwargs=None):
        B = xt.shape[0]
        out = self.p_mean_variance(xt, t, model_kwargs=model_kwargs) 
        '''
        stage1: guide mean
        stage2 :replace mean
        '''
        if self.modeltype == 's1': # Spatial Guidance
            out['mean'] = self.lbfgs_guide2(out['mean'], t, model_kwargs)
        elif self.modeltype == 's2':
            if partial_emb is not None: 
                out['mean'] = torch.where(model_kwargs['traj_mask_263'], partial_emb, out['mean']) 
        else:
            raise NotImplementedError

        mean = out['mean']   
        var = out['variance']
        log_var = out['log_variance']
        pred_x0 = out['pred_x0']

        noise = torch.randn_like(xt)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1))) # no noise when t == 0
        sample = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
            
        return {"sample": sample, # x_{t-1}
                "pred_x0": pred_x0}
    
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


        if self.modeltype == 's1': # Spatial Guidance
            mean_pred = self.lbfgs_guide2(mean_pred, t, model_kwargs)
        if self.modeltype == 's2':
            mean_pred = torch.where(model_kwargs['traj_mask_263'], partial_emb, mean_pred) 

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(xt.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise


        return {"sample": sample,
                "pred_x0": out['pred_x0']}


    def p_mean_variance(self, masked_xt, t, model_kwargs=None):
        B = masked_xt.shape[0]
        assert t.shape == (B,)
        
        traj = model_kwargs['traj']
        clip_text = model_kwargs['clip_text']
        
        assert masked_xt.shape[0] == len(clip_text) == traj.shape[0]
        if self.args.use_ddim:   # 2024-11-20添加，用于DDIM
            sample_t = self.timestep_map[t]
        else:
            sample_t = t.clone()
            
        scale = torch.ones(B,device=torch.device('cuda')) * self.args.guidance_param
        y = {'text':clip_text, 'scale':scale,}
        if self.modeltype == 's1':
            xt = masked_xt.permute(0,2,1)[:,:,None]
            if y.get('text_emb') is not None:
                y['text_emb'] = y.get('text_emb')
            y['hint'] = traj.flatten(2,3)
            pred_x0 = self.model(xt, sample_t, y=y)
            pred_x0 = pred_x0.squeeze(2).permute(0,2,1)
        elif self.modeltype == 's2':
            xt = masked_xt.permute(0,2,1)[:,:,None]
            if y.get('text_emb') is not None:
                y['text_emb'] = y.get('text_emb')
            y['traj_mask_263'] = model_kwargs['traj_mask_263'].float()
            pred_x0 = self.model(xt, sample_t, y=y)  # (b,196,263)
            pred_x0 = pred_x0.squeeze(2).permute(0,2,1)
        else:
            raise NotImplementedError
        

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        model_variance = _extract_into_tensor(model_variance, t, masked_xt.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, masked_xt.shape)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_x0, x_t=masked_xt, t=t) 
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

        joint_pos = recover_from_ric(x_, n_joints)
        if n_joints == 21:
            joint_pos = joint_pos * 0.001
            hint = hint * 0.001

        loss = torch.norm((joint_pos - hint) * mask_hint, dim=-1)
        return loss
    
    def lbfgs_guide2(self, x, t, condition):
        
        sample_t = t
        n_joint = 22 if self.args.dataset_name == 't2m' else 21
        
        if self.model.training:
            n_guide_steps = 10
        else:
            if sample_t[0] >= 10:
                n_guide_steps = 1 # 10 ~ 999
            elif sample_t[0] < 10 and sample_t[0] >0:
                n_guide_steps = 10  # 1 ~ 9
            else:
                n_guide_steps = 100 # 0

        lrs = np.linspace(self.args.bfgs_lr,0.1,1000)
        lr = lrs[sample_t[0]]
        
        mask_hint = condition['traj_mask']
        hint = condition['traj'].clone().detach()
        
        if self.raw_std.device != hint.device:
            self.raw_mean = self.raw_mean.to(hint.device)
            self.raw_std = self.raw_std.to(hint.device)
            self.mean = self.mean.to(hint.device)
            self.std = self.std.to(hint.device)
        hint = hint * self.raw_std + self.raw_mean
        hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint  

        with torch.enable_grad():
            x = x.clone().detach().contiguous().requires_grad_(True)

            def closure():
                lbfgs.zero_grad()
                loss = self.masked_joint_loss(x, self.mean, self.std, hint, mask_hint).mean()
                loss.backward()
                return loss

            lbfgs = torch.optim.LBFGS([x],
                        history_size=10, 
                        max_iter=10*n_guide_steps if not self.model.training else 10,
                        lr = lr,
                        tolerance_change=1e-8,
                        line_search_fn="strong_wolfe")
                
            lbfgs.step(closure)
        return x
    

    def omnicontrol_guide(self, x, t, condition, t_stopgrad=-10, scale=.5, n_guide_steps=10, train=False, min_variance=0.01):
        """
        Spatial guidance
        """
        sample_t = self.timestep_map[t] if self.args.use_ddim else t
        n_joint = 22 if self.args.dataset_name == 't2m' else 21
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
        

        mask_hint = condition['traj_mask']
        # mask_hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3).sum(dim=-1, keepdim=True) != 0
        hint = condition['traj'].clone().detach()
        if self.raw_std.device != hint.device:
            self.raw_mean = self.raw_mean.to(hint.device)
            self.raw_std = self.raw_std.to(hint.device)
            self.mean = self.mean.to(hint.device)
            self.std = self.std.to(hint.device)
        hint = hint * self.raw_std + self.raw_mean
        hint = hint.view(hint.shape[0], hint.shape[1], n_joint, 3) * mask_hint

        
        if not train:
            scale = self.calc_grad_scale(mask_hint[..., :1])


        for _ in range(n_guide_steps):
            loss, grad = self.gradients(x, self.mean, self.std, hint, mask_hint, condition['real_length'])

            grad = model_variance * grad
            if t[0] >= t_stopgrad:
                x = x - scale * grad
        return x.detach()
    
    def calc_grad_scale(self, mask_hint):
        assert mask_hint.shape[1] == 196
        num_keyframes = mask_hint.sum(dim=1).squeeze(-1)
        max_keyframes = num_keyframes.max(dim=1)[0]
        scale = 20 / max_keyframes
        return scale.unsqueeze(-1).unsqueeze(-1)

    def gradients(self, x, mean, std, hint, mask_hint, real_length, joint_ids=None):
        with torch.enable_grad():
            x.requires_grad_(True)
            b = x.shape[0]
            dim = x.shape[-1]
            x_ = x * std[..., :dim] + mean[..., :dim]
            n_joints = 22 if dim in [67, 193, 259, 263] else 21
            joint_pos = recover_from_ric(x_, n_joints) # (b,196,22,3)
            if n_joints == 21:
                joint_pos = joint_pos * 0.001
                hint = hint * 0.001

            loss = torch.norm((joint_pos - hint) * mask_hint, dim=-1)
            grad = torch.autograd.grad([loss.sum()], [x])[0] # （b, l, 67）
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



def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

