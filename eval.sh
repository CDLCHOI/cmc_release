./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth diffmdm output/0816_diffmdm_bfgs2/net_last.pth "0 10 11 15 20 21" 3 both_text_spatial sample 1 1 1000 1000 0 0.5 2 # 使用BFGS2

#####################################################################
#####################################################################
# HumanML3D 主表格
# 原目录里的 0816_diffmdm_bfgs2
python eval_cmc.py --roottype omni67mdm_spatial --resume_root output/1009_omni67mdm_spatial_bfgs/net_last.pth\
 --modeltype diffmdm --resume_trans output/0816_diffmdm_bfgs2/net_last.pth\
 --control_joint 0 --density 100 --gpu 7\
 --use_lbfgs 1 --replication_times 1 --S1_diffusion_step 1000 --S2_diffusion_step 1000\
 --only_t2m_s2 0 --bfgs_lr 0.5 --bfgs_type 2
# DDIM 100步
python eval_cmc.py --roottype omni67mdm_spatial --resume_root output/1009_omni67mdm_spatial_bfgs/net_last.pth\
 --modeltype diffmdm --resume_trans output/0816_diffmdm_bfgs2/net_last.pth\
 --control_joint 21 --density 100 --gpu 3\
 --use_lbfgs 1 --replication_times 1 --S1_diffusion_step 100 --S2_diffusion_step 100\
 --only_t2m_s2 0 --bfgs_lr 0.5 --bfgs_type 2 --use_ddim

# HumanML3D text to motion  和带轨迹控制的命令上的区别就是直接把root的type和ckpt不赋值，默认就是None
python eval_cmc.py --modeltype diffmdm --resume_trans output/0816_diffmdm_bfgs2/net_last_.pth\
 --gpu 3 --replication_times 10 --S1_diffusion_step 1000 --S2_diffusion_step 1000\
 --only_t2m_s2 1

./eval_diffmdm.sh omni67mdm_spatial output/1028_omnimdm_spatial_bfgs07_kit_control0/net_last.pth diffmdm output/1025_diffmdm_kit/net_last.pth 0 0 both_text_spatial sample 1 1 1000 1000 0 0.7 0

#####################################################################
#####################################################################
# KIT 主表格
# 原目录里的 output/1025_diffmdm_kit_bfgs2_论文表格
python eval_cmc.py --roottype omni67mdm_spatial --resume_root output/1028_omnimdm_spatial_bfgs07_kit/net_last.pth\
 --modeltype diffmdm --resume_trans output/1025_diffmdm_kit_bfgs2/net_last.pth --dataset_name kit\
 --control_joint 0 21 --density 100 --gpu 3\
 --use_lbfgs 1 --replication_times 1 --S1_diffusion_step 1000 --S2_diffusion_step 1000\
 --only_t2m_s2 0 --bfgs_lr 0.5 --bfgs_type 2
# DDIM 
python eval_cmc.py --roottype omni67mdm_spatial --resume_root output/1028_omnimdm_spatial_bfgs07_kit/net_last.pth\
 --modeltype diffmdm --resume_trans output/1025_diffmdm_kit_bfgs2/net_last.pth --dataset_name kit\
 --control_joint 0 21 --density 100 --gpu 3\
 --use_lbfgs 1 --replication_times 1 --S1_diffusion_step 100 --S2_diffusion_step 100\
 --only_t2m_s2 0 --bfgs_lr 0.5 --bfgs_type 2 --use_ddim


# KIT text to motion 和带轨迹控制的命令上的区别就是直接把root的type和ckpt不赋值，默认就是None
python eval_cmc.py --modeltype diffmdm --resume_trans output/1025_diffmdm_kit/net_last.pth\
 --dataset_name kit --gpu 4 --replication_times 10 --S1_diffusion_step 1000 --S2_diffusion_step 1000\
 --only_t2m_s2 1


########################## CMC release
python eval_cmc.py --resume_root output/cmc_humanml3d/cmc_humanml3d_s1.pth --resume_trans output/cmc_humanml3d/cmc_humanml3d_s2.pth --dataset_name t2m --control_joint 0 --density 100 --gpu 3 --use_lbfgs 1 --replication_times 1 --S1_diffusion_step 100 --S2_diffusion_step 100 --only_t2m_s2 0 --bfgs_lr 0.5 --bfgs_type 2 --use_ddim 1

python eval_cmc.py --roottype omni67mdm_spatial --resume_root output/cmc_kit/cmc_kit_s1.pth --modeltype diffmdm --resume_trans output/cmc_kit/cmc_kit_s2.pth --dataset_name kit --control_joint 0 --density 100 --gpu 3 --use_lbfgs 1 --replication_times 1 --S1_diffusion_step 100 --S2_diffusion_step 100 --only_t2m_s2 0 --bfgs_lr 0.5 --bfgs_type 2 --use_ddim 1