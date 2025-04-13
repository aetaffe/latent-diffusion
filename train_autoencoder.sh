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
python main.py --base configs/autoencoder/autoencoder_vq_cholec_64x64x4.yaml -t --gpus 1