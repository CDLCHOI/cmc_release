import torch
from data_loaders.humanml.networks.modules import *
from torch.utils.data import Dataset
from utils.mask_utils import generate_src_mask, calc_loss_xyz, calc_loss_xyz_perbatch

class CompADCGeneratedDataset(Dataset):
    def __init__(self, args, gen_loader, clip_model, diffusion_root, diffusion, mm_num_samples, mm_num_repeats, num_samples_limit):
        self.args = args
        self.gen_loader = gen_loader
        self.dataset = gen_loader.dataset
        assert mm_num_samples < len(gen_loader.dataset)
        num_samples_limit = len(self.dataset) if num_samples_limit > len(self.dataset) else num_samples_limit
        real_num_batches = len(gen_loader)
        if num_samples_limit is not None:
            real_num_batches = num_samples_limit // gen_loader.batch_size + 1
        print('real_num_batches', real_num_batches)


        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // gen_loader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs = ', mm_idxs)
        # samples = []
        # gt_motions = []
        # real_lengths = []
        for i, batch in enumerate(self.gen_loader):
            print(f'{i}/{real_num_batches}')
            if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                break
            
            word_embeddings, pos_one_hots, clip_text, sent_len, gt_motion, real_length, txt_tokens, traj, traj_mask_263, traj_mask, filename = batch
            txt_tokens = [t.split('_') for t in txt_tokens]
            b, max_length, num_features = gt_motion.shape
            gt_motion = gt_motion.cuda()
            real_length = real_length.cuda()
            # real_lengths.append(real_length)
            traj = traj.cuda()
            traj_mask = traj_mask.cuda()
            traj_mask_263 = traj_mask_263.cuda()
            real_mask = generate_src_mask(max_length, real_length) # (b,196)
            gt_ric = gt_motion[..., :67]
            
            #encode text
            # text_emb = diffusion.model.model.encode_text(clip_text)

            if args.stage2_no_root_y and (0 not in args.control_joint):
                traj_mask_263[..., 3] = False

            model_kwargs = {}
            model_kwargs['traj'] = traj.clone()
            # model_kwargs['text_emb'] = text_emb
            # model_kwargs['word_emb'] = word_emb
            model_kwargs['traj_mask'] = traj_mask
            model_kwargs['traj_mask_263'] = traj_mask_263
            model_kwargs['gt_motion'] = gt_motion
            model_kwargs['real_mask'] = real_mask
            model_kwargs['clip_text'] = clip_text
            model_kwargs['real_length'] = real_length

            # ipdb.set_trace()


            is_mm = i in mm_idxs 
            repeat_times = mm_num_repeats if is_mm else 1
            mm_motions = []

            ########################################################
            ########################  1阶段 ########################
            ########################################################
            if not args.only_t2m_s2:
                if args.roottype in ['omni67mdm_spatial']: 
                    if args.gtric_fortest:
                        pred_ric = gt_ric
                    else:
                        pred_ric = diffusion_root.p_sample_loop(partial_emb=None, model_kwargs=model_kwargs,batch_size=args.batch_size)
                    
                    # control_id = traj_mask[0].sum(0).sum(-1).nonzero()
                    # if args.roottype == 'omnicontrol':
                    #     sample = pred_ric
                else:
                    raise NotImplementedError
            

            ########################################################
            ########################  2阶段 ########################  
            ########################################################
            for t in range(repeat_times):
                if args.only_t2m_s2: # 仅2阶段text to motion
                    partial_emb = None
                else:
                    partial_emb = torch.zeros_like(gt_motion, device=gt_motion.device)
                    if self.args.dataset_name == 't2m':
                        partial_emb[..., :67] = pred_ric[..., :67]  
                    else:
                        partial_emb[..., :64] = pred_ric[..., :64]  
                sample = diffusion.p_sample_loop(partial_emb, with_control=True, model_kwargs=model_kwargs, batch_size=args.batch_size) # (b, 196, 263)
            

                ########################################################
                ########################################################
                ########################################################

                if t == 0:
                    sub_dicts = [{'motion': sample[bs_i].squeeze().cpu().numpy(),
                                'length': real_length[bs_i].cpu().numpy(),
                                'caption': clip_text[bs_i],
                                'hint': traj[bs_i].cpu().numpy(),
                                'tokens': txt_tokens[bs_i],
                                'cap_len': sent_len[bs_i].item(),
                                'filename': filename[bs_i],
                                } for bs_i in range(gen_loader.batch_size)]
                    
                    generated_motion += sub_dicts
                    
                if is_mm:
                    mm_motions += [{'motion': sample[bs_i].squeeze().cpu().numpy(),
                                    'length': real_length[bs_i].cpu().numpy(),
                                    } for bs_i in range(gen_loader.batch_size)]

            
            if is_mm:
                mm_generated_motions += [{
                                'caption': clip_text[bs_i],
                                'tokens': txt_tokens[bs_i],
                                'cap_len': sent_len[bs_i].item(),
                                'mm_motions': mm_motions[bs_i::gen_loader.batch_size], 
                                } for bs_i in range(gen_loader.batch_size)]
            a = 1
        
        self.generated_motion = generated_motion
        self.mm_generated_motion = mm_generated_motions
        self.w_vectorizer = gen_loader.dataset.w_vectorizer

        self.eval_mean = np.load('/home/deli/project/MARDM/utils/eval_mean_std/t2m/eval_mean.npy')
        self.eval_std = np.load('/home/deli/project/MARDM/utils/eval_mean_std/t2m/eval_std.npy')

    def __len__(self):
        return len(self.generated_motion)
    
    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens, hint, filename = data['motion'], data['length'], data['caption'], data['tokens'], data['hint'], data['filename']
        sent_len = data['cap_len']

        if self.dataset.mode == 'eval':
            
            normed_motion = motion
            denormed_motion = self.dataset.t2m_dataset.inv_transform(normed_motion)
            renormed_motion = (denormed_motion - self.dataset.mean_for_eval) / self.dataset.std_for_eval  # according to T2M norms
            motion = renormed_motion
            # This step is needed because T2M evaluators expect their norm convention

        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), hint, filename
    