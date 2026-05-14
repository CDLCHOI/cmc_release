##### 同时输出控制指标和动作指标

# 1. omni67mdm_spatial
# 2. diffmdm
# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth diffmdm output/0816_diffmdm/net_last.pth 0 4 both_text_spatial 1 1 1000 1000 0 0.5 2
# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth diffmdm output/0816_diffmdm/net_last.pth 0 7 both_text_spatial 1 20 1000 1000 1 0.5 0  # 纯文本 mm
# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth diffmdm output/0816_diffmdm_bfgs2/net_last.pth "0 10 11 15 20 21" 3 both_text_spatial 1 1 1000 1000 0 0.5 2 # 使用BFGS2

# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth semboost output/0520_semboost_bfgs2/net_last.pth "0 10 11 15 20 21" 4 both_text_spatial 1 1 1000 1000 0 0.5 2

# 验omnicontrol bfgs
# ./eval_diffmdm.sh omnicontrol output/omnicontrol_ckpt/model_humanml3d.pt diffmdm output/0816_diffmdm/net_last.pth 0 3 both_text_spatial 1 1 1000 1000 0 0.5 2  # 有bfgs
# ./eval_diffmdm.sh omnicontrol output/omnicontrol_ckpt/model_humanml3d.pt diffmdm output/0816_diffmdm/net_last.pth 0 3 both_text_spatial 0 1 1000 1000 0 0.5 2  # SGD

# 1. omni67mdm_spatial
# 2. diffmdm_mask
# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth diffmdm_mask_res output/0912_diffmdm_mask/net_last.pth 0 5 both_text_spatial priorMDM 1 1 1000 1000 0 0.5

# 1. omni67mdm_spatial
# 2. diffmdm_mask_res
# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth diffmdm_mask_res output/0912_diffmdm_mask_res/net_last.pth 0 5 both_text_spatial priorMDM 1 1 1000 1000 0 0.5

# 1. omni67mdm_spatial
# 2. omnimdm
# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth omnimdm output/1010_omnimdm/net_last.pth 0 3 both_text_spatial 1 1 1000 1000 0 0.5
# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth omnimdm output/1010_omnimdm_only_inpaint/net_last.pth 0 4 both_text_spatial priorMDM 1 1 1000 1000 0 0.5

# 1. omni67mdm_spatial
# 2. omnimdm_mask
# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth omnimdm_mask output/0918_omnimdm_mask_only_inpaint/net_last.pth 0 0 both_text_spatial priorMDM 1 1 1000 1000 0 0.5

############# KIT
# ./eval_diffmdm.sh omni67mdm_spatial output/1028_omnimdm_spatial_bfgs07_kit_control0/net_last.pth diffmdm output/1025_diffmdm_kit/net_last.pth 0 0 both_text_spatial 1 1 1000 1000 0 0.7 0
# ./eval_diffmdm.sh omni67mdm_spatial output/1028_omnimdm_spatial_bfgs07_kit/net_last.pth diffmdm output/1025_diffmdm_kit/net_last.pth 0 5 both_text_spatial 1 20 1000 1000 1 0.7 0  #  纯文本 only_t2m_s2

# ./eval_diffmdm.sh omni67mdm_spatial output/1028_omnimdm_spatial_bfgs07_kit_control0/net_last.pth diffmdm output/1025_diffmdm_kit_bfgs1/net_last.pth 0 1 both_text_spatial 1 1 1000 1000 0 0.3 1

# ./eval_diffmdm.sh omni67mdm_spatial output/1028_omnimdm_spatial_bfgs07_kit_control0/net_last.pth diffmdm output/1025_diffmdm_kit_bfgs2/net_last.pth 0 2 both_text_spatial 1 1 1000 1000 0 0.5 2

# 消融 单阶段 BFGStype0
# ./eval_diffmdm.sh omni263mdm_spatial /home/deli/project/ADControl/output/0926_omni263mdm_spatial_bfgs/net_last.pth diffmdm output/0816_diffmdm_bfgs2/net_last.pth 0 2 both_text_spatial 1 1 1000 1000 0 0.5 0
# 消融 不分阶段普通guide 20
# ./eval_diffmdm.sh omni263mdm_spatial /home/deli/project/ADControl/output/0926_omni263mdm_spatial_bfgs/net_last.pth diffmdm output/0816_diffmdm_bfgs2/net_last.pth 20 3 both_text_spatial 0 1 1000 1000 0 0.5 0

# 2025-03-18 测一下原本的MDM
# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth diffmdm output/humanml_trans_enc_512/model000475000.pt 0 4 both_text_spatial 1 20 1000 1000 1 0.5 0
# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth diffmdm output/0318_diffmdm/net_last.pth 0 4 both_text_spatial 1 20 1000 1000 1 0.5 0 # resample_mask的测一下
# ./eval_diffmdm.sh omni67mdm_spatial output/1028_omnimdm_spatial_bfgs07_kit/net_last.pth diffmdm output/0319_diffmdm_kit/net_last.pth 0 5 both_text_spatial 1 20 1000 1000 1 0.7 0


roottype=$1
root_path=$2
modeltype=$3
resume_trans=$4
joint=$5
gpu=$6
cond_mode=$7
use_lbfgs=$8
repeat_time=$9
S1_diffusion_step=${10}
S2_diffusion_step=${11}
only_t2m_s2=${12}


python eval_ADControl.py --roottype ${roottype} --resume_root ${root_path} --modeltype ${modeltype} --resume_trans ${resume_trans} --control_joint ${joint} --density 100 --gpu ${gpu} --use_lbfgs ${use_lbfgs} --replication_times ${repeat_time} --S1_diffusion_step ${S1_diffusion_step} --S2_diffusion_step ${S2_diffusion_step} --only_t2m_s2 ${only_t2m_s2}

# python eval_ADControl.py --roottype omni67mdm_spatial --resume_root output/1009_omni67mdm_spatial_bfgs/net_last.pth --modeltype diffmdm --resume_trans output/0816_diffmdm_bfgs3/net_last.pth --control_joint 10 --density 1 --gpu 1 --use_lbfgs 1 --replication_times 1 --S1_diffusion_step 1000 --S2_diffusion_step 1000 --only_t2m_s2 0

# ./eval_diffmdm.sh omni67mdm_spatial output/1009_omni67mdm_spatial_bfgs/net_last.pth diffmdm output/0816_diffmdm/net_last.pth 0 4 both_text_spatial 1 1 1000 1000 0
