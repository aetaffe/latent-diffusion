#!/bin/bash -l
#SBATCH -J ldm-train
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=13-00:00:00
#SBATCH --partition=gpu-qi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aetaffe@ucdavis.edu
#SBATCH -o jobs/train_ldm-%j.output
#SBATCH -e jobs/train_ldm-%j.error
module load conda3/4.X
conda activate stylegan3
python main.py --base configs/latent-diffusion/flim-ldm-vq-4-hpc.yaml -t --gpus 1 --no-test \
--resume /home/ataffe/SyntheticData/latent-diffusion/logs/2025-04-15T05-00-07_flim-ldm-vq-4-hpc/checkpoints/last.ckpt