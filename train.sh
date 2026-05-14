论文名称：Coordinating Multiple Conditions for Trajectory-Controlled Human Motion Generation
主页路径：https://cdlchoi.github.io/cmc_page/
arxiv路径：https://arxiv.org/abs/2605.13729

准备阶段-配置conda环境：
    conda env create -f environment.yml
    pip install git+https://github.com/openai/CLIP.git
    按照 SMPL仓库 https://github.com/vchoutas/smplx 安装 SMPLX

准备阶段-下载预训练权重：
    从googledrive链接下载HumanML3D和KIT数据集的权重，解压到output/目录，应有如下的文件结构：
    ./output/cmc_humanml3d/cmc_humanml3d_s1.pth
    ./output/cmc_humanml3d/cmc_humanml3d_s2.pth
    ./output/cmc_kit/cmc_kit_s1.pth
    ./output/cmc_kit/cmc_kit_s2.pth


Train your own model:
# Training CMC HumanML3D stage1
python train.py --exp_name cmc_humanml3d_s1 --batch_size 128 --gpu 2 --overwrite --print_iter 50 --save_iter 10000 --total_iter 600000 --lr 1e-4 --lr-scheduler 300000 --modeltype s1 --multi_joint_control
# Training CMC HumanML3D stage2
python train.py --exp_name cmc_humanml3d_s2 --batch_size 128 --gpu 2 --overwrite --print_iter 100 --save_iter 10000 --total_iter 600000 --lr 1e-4 --lr-scheduler 300000 --modeltype s2 --multi_joint_control

Evaluation:
# HumanML3D s1+s2 DDPM 1000+1000
# For multi-joint control, e.g., --control_joint 0 20
python eval_cmc.py --resume_root output/cmc_humanml3d/cmc_humanml3d_s1.pth --resume_trans output/cmc_humanml3d/cmc_humanml3d_s2.pth --dataset_name t2m --control_joint 0 --density 100 --gpu 3

# HumanML3D s1+s2 DDIM 100+100
python eval_cmc.py --resume_root output/cmc_humanml3d/cmc_humanml3d_s1.pth --resume_trans output/cmc_humanml3d/cmc_humanml3d_s2.pth --dataset_name t2m --control_joint 0 --density 100 --gpu 3 --S1_diffusion_step 100 --S2_diffusion_step 100

# HumanML3D only s2 text-to-motion
python eval_cmc.py --resume_trans output/cmc_humanml3d/cmc_humanml3d_s2.pth --dataset_name t2m --gpu 3 --only_t2m_s2 1

# HumanML3D sample
python sample.py
