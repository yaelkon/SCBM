#!/bin/bash

#SBATCH --job-name=SCBM
#SBATCH --output="/path/to/output" 
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1

source ~/.bashrc
conda deactivate
conda activate scbm
cd /home/yk449/python_projects/SCBM

python -u train.py "$@"