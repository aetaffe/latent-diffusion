#!/bin/bash -l
#SBATCH -J vqvae-train
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=13-00:00:00
#SBATCH --partition=gpu-qi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aetaffe
#SBATCH -o jobs/train_autoencoder-%j.output
#SBATCH -e jobs/train_autoencoder-%j.error
module load conda3/4.X
conda activate stylegan3
python main.py --base configs/autoencoder/autoencoder_vq_flim-fine-tune_16x16x256.yaml -t --no-test \
 --resume /home/ataffe/SyntheticData/latent-diffusion/logs/2025-04-09T04-49-29_autoencoder_vq_cholec_64x64x4/checkpoints/epoch=000909.ckpt