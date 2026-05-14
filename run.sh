OMP_NUM_THREADS=8 python train.py --exp_name 0514_mdm --batch_size 128 --gpu 2 --overwrite --print_iter 100 --save_iter 10000 --total_iter 600000 --lr 2e-4 --lr-scheduler 250000 --modeltype mdm

OMP_NUM_THREADS=8 python train.py --exp_name 0514_mdm_sim --batch_size 128 --gpu 3 --overwrite --print_iter 100 --save_iter 10000 --total_iter 600000 --lr 2e-4 --lr-scheduler 250000 --modeltype mdm --sim


OMP_NUM_THREADS=8 python train.py --exp_name 0519_mdm_el50 --batch_size 128 --gpu 1 --overwrite --print_iter 100 --save_iter 10000 --total_iter 500000 --lr 2e-4 --lr-scheduler 200000 --modeltype mdm --emb_loss 50 # √√√√√√√√√  好
OMP_NUM_THREADS=8 python train.py --exp_name 0521_mdm_el1 --batch_size 128 --gpu 5 --overwrite --print_iter 100 --save_iter 10000 --total_iter 300000 --lr 2e-4 --lr-scheduler 100000 --modeltype mdm --emb_loss 1 --eval_during_train # emb_loss系数设为1

OMP_NUM_THREADS=8 python train.py --exp_name 0519_mdm_only_el50 --batch_size 128 --gpu 2 --overwrite --print_iter 100 --save_iter 10000 --total_iter 500000 --lr 2e-4 --lr-scheduler 200000 --modeltype mdm --emb_loss 50 --only_emb_loss

python eval_cmc.py --modeltype mdm --resume_trans output/0521_mdm_el1/net_last.pth\
 --gpu 2 --replication_times 5 --S1_diffusion_step 1000 --S2_diffusion_step 1000\
 --only_t2m_s2 1 --max_samples 1000


######################################################################################

OMP_NUM_THREADS=8 python train.py --exp_name 0519_critic --batch_size 32 --gpu 4 5 --overwrite --print_iter 100 --save_iter 10000 --total_iter 200000 --lr 2e-4 --lr-scheduler 100000 --modeltype critic --use_cache # 32 batch，要用2张卡太大了  还没跑

OMP_NUM_THREADS=8 python train.py --exp_name 0519_mdmcritic --batch_size 128 --gpu 3 --overwrite --print_iter 100 --save_iter 10000 --total_iter 500000 --lr 2e-4 --lr-scheduler 200000 --modeltype mdmcritic --use_cache --num_noisy_timesteps 100

OMP_NUM_THREADS=8 python train.py --exp_name 0519_mdmcritic_hml --batch_size 128 --gpu 3 --overwrite --print_iter 100 --save_iter 10000 --total_iter 500000 --lr 2e-4 --lr-scheduler 200000 --modeltype mdmcritic --use_cache --num_noisy_timesteps 100 --datatype hml
