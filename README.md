
# Coordinating Multiple Conditions for Trajectory-Controlled Human Motion Generation (TMM 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2605.13729-B31B1B.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2605.13729)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg?style=flat-square)](https://cdlchoi.github.io/cmc_page/)

This is the official PyTorch implementation for the paper **"Coordinating Multiple Conditions for Trajectory-Controlled Human Motion Generation"**.

---

## 🛠 Preparation

### 1. Configure Conda Environment
We recommend using Conda to set up the environment. Run the following commands:

```bash
conda env create -f environment.yml
conda activate cmc

pip install git+https://github.com/openai/CLIP.git

```


### 2. Download Pre-trained Weights

Download the pre-trained weights for the HumanML3D and KIT datasets from our [Google Drive link](https://drive.google.com/drive/folders/19iGUkZ7hK3W0_Hn_7un9CQ9iE1-2SyMY?usp=drive_link).

Extract the downloaded files into the `output/` directory. The file structure should look exactly like this:

```text
./output/
├── cmc_humanml3d/
│   ├── cmc_humanml3d_s1.pth
│   └── cmc_humanml3d_s2.pth
└── cmc_kit/
    ├── cmc_kit_s1.pth
    └── cmc_kit_s2.pth

```

### 3. Download the evaluator

Follow [MoMask Github](https://github.com/EricGuo5513/momask-codes). The structure should look like this:

```text
./checkpoints/
├── t2m/
│   ├── text_mot_match
└── kit/
    ├── text_mot_match

```

### 4. Soft link to your dataset path

```bash
ln -s /path/to/dataset/HumanML3D/ dataset/HumanML3D/
ln -s /path/to/dataset/KIT-ML/ dataset/KIT-ML/
```

---

## 🚀 Training

To train your own model from scratch, execute the following commands.

**Training CMC HumanML3D Stage 1:**

```bash
python train.py \
    --exp_name cmc_humanml3d_s1 \
    --batch_size 128 --gpu 2 --overwrite --save_iter 10000 \
    --total_iter 600000 --lr 1e-4 --lr-scheduler 300000 \
    --modeltype s1 --multi_joint_control

```

**Training CMC HumanML3D Stage 2:**

```bash
python train.py \
    --exp_name cmc_humanml3d_s2 \
    --batch_size 128 --gpu 2 --overwrite --save_iter 10000 \
    --total_iter 600000 --lr 1e-4 --lr-scheduler 300000 \
    --modeltype s2 --multi_joint_control

```

---

## 📊 Evaluation

We provide several evaluation scripts for different configurations. For multi-joint control, you can specify joints using `--control_joint` (e.g., `--control_joint 0 20`).

**1. HumanML3D (Stage 1 + Stage 2) | DDPM 1000+1000 steps:**

```bash
python eval_cmc.py \
    --resume_root output/cmc_humanml3d/cmc_humanml3d_s1.pth \
    --resume_trans output/cmc_humanml3d/cmc_humanml3d_s2.pth \
    --dataset_name t2m \
    --control_joint 0 --density 100 --gpu 3

```

**2. HumanML3D (Stage 1 + Stage 2) | DDIM 100+100 steps:**

```bash
python eval_cmc.py \
    --resume_root output/cmc_humanml3d/cmc_humanml3d_s1.pth \
    --resume_trans output/cmc_humanml3d/cmc_humanml3d_s2.pth \
    --dataset_name t2m \
    --control_joint 0 --density 100 --gpu 3 \
    --S1_diffusion_step 100 --S2_diffusion_step 100

```

**3. HumanML3D (Stage 2 Only) | Text-to-Motion:**

```bash
python eval_cmc.py \
    --resume_trans output/cmc_humanml3d/cmc_humanml3d_s2.pth \
    --dataset_name t2m \
    --gpu 3 \
    --only_t2m_s2 1
```

**4. HumanML3D Sample:**

```bash
python sample.py
```
This script will visualize motion in a web page and save html file.
