
# HumanML3D 2阶段
OMP_NUM_THREADS=8 python train.py --exp_name 0816_diffmdm --batch_size 128 --gpu 2 --overwrite --print_iter 50 --save_iter 10000 --total_iter 200000 --lr 1e-5 --lr-scheduler 160000 --modeltype diffmdm --loss_xyz 1 --loss_type l2 --multi_joint_control --root_dist_loss --resume_trans output/0519_diffmdm/net_last.pth
# HumanML3D 1阶段
OMP_NUM_THREADS=8 python train.py --exp_name 1009_omni67mdm_spatial_bfgs --batch_size 128 --gpu 2 --overwrite --print_iter 50 --save_iter 10000 --total_iter 350000 --lr 2e-4 --lr-scheduler 100000 250000 --modeltype omni67mdm_spatial --loss_xyz 1 --loss_type l2 --normalize_traj --multi_joint_control --root_dist_loss --resume_trans /home/deli/project/ADControl/output/0826_omni67mdm_spatial_bfgs10/net_last.pth --use_lbfgs 1