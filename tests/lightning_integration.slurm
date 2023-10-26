#!/bin/bash -l
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=18
# #SBATCH --mem-per-cpu=2GB
# #SBATCH --constraint="gpu"
# #SBATCH --gres=gpu:a100:4

python lightning_integration.py
