#!/bin/bash
#SBATCH --partition=a100
#SBATCH --job-name=test_script
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # change this if you want more than 1 gpu
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --mail-user=cq2u24@soton.ac.uk

module purge
eval "$(conda shell.bash hook)"
module load conda

conda activate imb

python main.py


