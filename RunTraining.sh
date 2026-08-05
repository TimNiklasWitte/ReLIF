#!/bin/bash

#SBATCH --job-name="run_job"
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

spack load cuda@11.8.0
spack load miniconda3

source ~/.bashrc conda

conda activate snntorch

dataset_name=$1
model_name=$2
freq_max=$3

nvidia-smi

python3 Training.py $dataset_name $model_name $freq_max

